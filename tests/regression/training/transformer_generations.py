"""
Train tiny GPT-2 for a short GrowingNN run on Tiny Shakespeare.

Downloads Karpathy's Tiny Shakespeare (~1 MB), tokenizes windows with the
tiny-GPT-2 tokenizer, feeds inputs_embeds into the HuggingFace FX graph, and
runs train_generations so MCTS can apply seq/res Linear actions.

Defaults: 8 generations, 10 epochs each, 256 train windows of length 16.
Expect about 5 to 15 minutes on CPU.

Run:
  python tests/regression/training/transformer_generations.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.score_functions.simulation_score import SimulationScore
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.trainer import train_generations
from growingnn.utils.fx.graph_extraction import extract_graph
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.regression.regression_utils import (
    FOLDER_NAME,
    log_action_count_table,
    parse_regression_cli,
)

MODEL_ID = "sshleifer/tiny-gpt2"
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
CACHE_DIR = "testResults/regression_cache"
TEXT_PATH = CACHE_DIR + "/tiny_shakespeare.txt"
OUT_DIR = FOLDER_NAME + "/transformer_train"
SEQ_LEN = 16
TRAIN_WINDOWS = 256
VAL_WINDOWS = 64
BATCH_SIZE = 16
GENERATIONS = 8
EPOCHS = 10
TEXT_CHARS = 40_000
SIMULATION_TIME = 30.0
SIMULATION_EPOCHS = 2
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")


def _download_tiny_shakespeare() -> str:
    """Return Tiny Shakespeare text, downloading once into regression_cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(TEXT_PATH):
        logger.info("downloading Tiny Shakespeare from %s", SHAKESPEARE_URL)
        urllib.request.urlretrieve(SHAKESPEARE_URL, TEXT_PATH)
    text = Path(TEXT_PATH).read_text(encoding="utf-8")
    logger.info("Tiny Shakespeare chars=%s cached=%s", len(text), TEXT_PATH)
    return text[:TEXT_CHARS]


def _token_windows(text: str, tokenizer, seq_len: int, n: int) -> list[list[int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows: list[list[int]] = []
    for start in range(0, len(ids) - seq_len, seq_len):
        chunk = ids[start : start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        windows.append(chunk)
        if len(windows) >= n:
            break
    return windows


def _embed_windows(
    windows: list[list[int]],
    embed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(windows, dtype=torch.long)
    tokens = ids[:, :-1]
    labels = ids[:, 1:]
    embeds = F.embedding(tokens, embed_weight)
    return embeds, labels


def _loaders(embed_weight: torch.Tensor, tokenizer) -> tuple[DataLoader, DataLoader]:
    text = _download_tiny_shakespeare()
    needed = TRAIN_WINDOWS + VAL_WINDOWS
    windows = _token_windows(text, tokenizer, SEQ_LEN, needed)
    if len(windows) < needed:
        logger.error("not enough token windows: got %s need %s", len(windows), needed)
        sys.exit(1)
    train_x, train_y = _embed_windows(windows[:TRAIN_WINDOWS], embed_weight)
    val_x, val_y = _embed_windows(windows[TRAIN_WINDOWS:needed], embed_weight)
    train = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE)
    logger.info(
        "windows train=%s val=%s seq_len=%s batch=%s x=%s y=%s",
        TRAIN_WINDOWS,
        VAL_WINDOWS,
        SEQ_LEN,
        BATCH_SIZE,
        tuple(train_x.shape),
        tuple(train_y.shape),
    )
    return train, val


def _plot_metric(values: list[float], name: str, save_path: str) -> None:
    steps = range(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, values)
    ax.set_xlabel("epoch (across generations)")
    ax.set_ylabel(name)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _write_summary(summary: dict, path: Path) -> None:
    epochs_per_gen = EPOCHS
    lines = [
        f"model={MODEL_ID}",
        f"dataset=Tiny Shakespeare ({TEXT_PATH})",
        f"seq_len={SEQ_LEN} train_windows={TRAIN_WINDOWS} val_windows={VAL_WINDOWS}",
        f"generations={GENERATIONS} epochs={EPOCHS} "
        f"sim_time={SIMULATION_TIME} sim_epochs={SIMULATION_EPOCHS}",
        "",
        "generation | action | last_train_acc | last_val_acc | param_count",
        "-----------+--------+----------------+--------------+------------",
    ]
    for i, gen in enumerate(summary["generation"]):
        epoch_idx = (i + 1) * epochs_per_gen - 1
        action = summary["generation_action"][i] or "(none)"
        lines.append(
            f"{gen} | {action} | {summary['train_acc'][epoch_idx]:.4f} | "
            f"{summary['val_acc'][epoch_idx]:.4f} | {summary['param_count'][epoch_idx]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("saved %s", path)


if __name__ == "__main__":
    parse_regression_cli()
    torch.manual_seed(0)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        logger.error("need transformers 4.x with utils.fx: %s", exc)
        sys.exit(1)
    logger.info("transformers %s", transformers.__version__)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.model_max_length = 100_000
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    model.config.use_cache = False
    n_embd = int(model.config.n_embd)
    embed_weight = model.transformer.wte.weight.detach().clone()
    logger.info("loaded %s n_embd=%s vocab=%s", MODEL_ID, n_embd, int(model.config.vocab_size))

    logger.info(
        "run gens=%s epochs=%s seq_len=%s train_windows=%s sim_time=%s sim_epochs=%s",
        GENERATIONS,
        EPOCHS,
        SEQ_LEN,
        TRAIN_WINDOWS,
        SIMULATION_TIME,
        SIMULATION_EPOCHS,
    )
    gm = extract_graph(model)
    train_loader, val_loader = _loaders(embed_weight, tokenizer)
    x0, _ = next(iter(train_loader))
    traced = TracedModel.create(gm, tuple(int(s) for s in x0[:1].shape))
    cfg_probe = RunningConfig(generations=1, epochs=1)
    cfg_probe.update_grow_actions(True)
    cfg_probe.update_shrink_actions(False)
    cfg_probe.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER = False
    cfg_probe.ACTIONS_ENABLE_ADD_RES_CONV_LAYER = False
    log_action_count_table(
        dict(Counter(type(a).__name__ for a in generate_all_actions(traced, cfg_probe))),
        title="actions before training",
    )

    draw_filtered_fx_graph(gm, OUT_DIR + "/fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, OUT_DIR + "/fx_graph0", fmt="pdf")

    cfg = RunningConfig(
        generations=GENERATIONS,
        epochs=EPOCHS,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        print_every=1,
        simulation_alg=montecarlo_alg,
        simulation_scheduler=AlwaysSimulationScheduler(
            simulation_time=SIMULATION_TIME, simulation_epochs=SIMULATION_EPOCHS
        ),
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
        simulation_set_size=32,
        criterion=nn.CrossEntropyLoss(),
        quiet=False,
    )
    cfg.update_shrink_actions(False)
    cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER = False
    cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER = False
    for flag in (
        "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01",
        "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_02",
        "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05",
    ):
        setattr(cfg, flag, False)

    sim_train = DataLoader(
        TensorDataset(*[t[:32] for t in train_loader.dataset.tensors]),
        batch_size=BATCH_SIZE,
    )
    sim_val = DataLoader(
        TensorDataset(*[t[:16] for t in val_loader.dataset.tensors]),
        batch_size=BATCH_SIZE,
    )

    gm, summary = train_generations(
        gm,
        train_loader,
        val_loader,
        cfg,
        sim_train_loader=sim_train,
        sim_val_loader=sim_val,
    )
    draw_filtered_fx_graph(gm, OUT_DIR + f"/fx_graph_simplified{GENERATIONS}", fmt="pdf")
    draw_torch_fx_graph(gm, OUT_DIR + f"/fx_graph{GENERATIONS}", fmt="pdf")

    os.makedirs(OUT_DIR, exist_ok=True)
    for key in METRIC_KEYS:
        _plot_metric(summary[key], key, OUT_DIR + f"/{key}.png")
    torch.save({k: summary[k] for k in (*METRIC_KEYS, "generation", "generation_action")}, OUT_DIR + "/history.pt")
    _write_summary(summary, Path(OUT_DIR) / "summary.txt")
    logger.info("generation_action=%s", summary["generation_action"])
    logger.info("done. last train_acc=%s val_acc=%s", summary["train_acc"][-1], summary["val_acc"][-1])
