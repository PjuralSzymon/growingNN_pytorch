"""Experiment 015: step-by-step redesign of the residual-block duplicate action on CIFAR-10.

Each step adds one change to the duplicate action, so every row can be read
against the row above it:

    current       quasi-identity copy, scale 1.0, copy before the ReLU, strict discovery, hard remove
    +side         block stays discoverable after AddResConvLayer wires an earlier layer into it
    +after_relu   copy is inserted after the block's ReLU
    +scale        copy's last branch layer scaled by --scale (default auto: the best
                  scale from the Part A sweep, so run Part A first; 0.3 without it)
    +merge        shrink folds a block into an identical neighbour instead of deleting it

Part A, paired checkpoints (about 1 hour): train a checkpoint once, apply
each setting to the same checkpoint, and train every copy for one generation.
No search is involved, so the rows show the action's own effect.

Part B, growing runs (about a day): the Experiment 014 CIFAR-10 protocol
(30 generations x 10 epochs, 8 s Monte Carlo search, 128 simulation
examples, 90% validation stop) for the "off" control and each step, 2 seeds.
Runs are ordered seed-first, so a complete seed-1 ladder exists after half
the time.

    pipenv run python -u experiments/train_cifar10_exp012_3_duplicate_action_design.py
    pipenv run python -u experiments/train_cifar10_exp012_3_duplicate_action_design.py --part a
    pipenv run python -u experiments/train_cifar10_exp012_3_duplicate_action_design.py --part b --seeds 1
    pipenv run python -u experiments/train_cifar10_exp012_3_duplicate_action_design.py --summary

Results: experiments/output/train_cifar10/runs/exp014_duplicate_action_design/
(summary.md is rewritten after every finished run and by --summary).

Historical record: after this experiment the action was simplified to one
version (copy after the ReLU, plain quasi-identity, side inputs discovered but
not copied, merge-or-remove shrink that keeps the last block). The design
switches this script compared were removed from growingnn, so only --summary
still runs; the code is kept to document how the results were produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_cifar10_exp009_best_reference import get_datasets
from experiments.train_cifar10_exp012_1_residual_block_action_grading import _claim_run_dir
from experiments.train_cifar10_exp012_2_residual_block_multidataset import (
    SIMULATION_EPOCHS,
    SIMULATION_SECONDS,
    SIMULATION_SET_SIZE,
    TARGET_VALIDATION_ACCURACY,
    ResidualStarter,
)
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.residual_block import ResidualBlockAction
from growingnn.board.experiment_board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

EXPERIMENT_ID = "exp014_duplicate_action_design"
OUT_DIR = ROOT / "experiments" / "output" / "train_cifar10" / "runs" / EXPERIMENT_ID
DATASET = "cifar10"
INPUT_SHAPE = (1, 3, 32, 32)
BATCH_SIZE = 128
DEFAULT_SEEDS = (1, 2)
DEFAULT_SCALE = 0.3
GENERATIONS = 30
EPOCHS = 10

# Part A training recipe (same optimizer settings as the growing runs).
CHECKPOINT_EPOCHS = 15
CHECKPOINT_LR = 0.05
GENERATION_EPOCHS = 10
GENERATION_LR = 0.02


def ladder(scale: float) -> list[tuple[str, dict | None]]:
    """Cumulative settings; None means residual-block actions are disabled."""
    current = dict(scale=1.0, after_activation=False, allow_side_inputs=False, shrink_mode="remove")
    side = {**current, "allow_side_inputs": True}
    after = {**side, "after_activation": True}
    scaled = {**after, "scale": scale}
    merged = {**scaled, "shrink_mode": "merge"}
    return [
        ("off", None),
        ("current", current),
        ("+side", side),
        ("+after_relu", after),
        ("+scale", scaled),
        ("+merge", merged),
    ]


def arm_folder(name: str, scale: float) -> str:
    return {"off": "0_off", "current": "1_current", "+side": "2_side", "+after_relu": "3_after_relu",
            "+scale": f"4_scale_{scale:g}", "+merge": f"5_merge_scale_{scale:g}"}[name]


def resolve_scale(requested: str) -> tuple[float, str]:
    """Return the scale for the +scale/+merge steps and where it came from.

    "auto" picks the scale with the best after-generation validation accuracy
    in Part A's scale sweep; without Part A results it falls back to 0.3.
    """
    if requested != "auto":
        value = float(requested)
        if not 0.0 <= value <= 1.0:
            raise ValueError("--scale must be between 0 and 1, or auto")
        return value, "set with --scale"
    path = OUT_DIR / "paired" / "paired_results.json"
    if path.is_file():
        tables = json.loads(path.read_text(encoding="utf-8"))["tables"]
        sweep = [r for r in tables.get("Scale sweep (side + after_relu), plain checkpoint", [])
                 if r.get("available") and r["setting"].startswith("scale ")]
        if sweep:
            best = max(sweep, key=lambda r: r["val_after_generation"])
            return float(best["setting"].split()[1]), "best of Part A scale sweep"
    return DEFAULT_SCALE, "default (no Part A results yet)"


# ---------------------------------------------------------------------------
# Part A: paired checkpoints
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train(model: nn.Module, loader, epochs: int, lr: float) -> None:
    device = _device()
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y, label_smoothing=0.1).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def _accuracy(model: nn.Module, loader) -> float:
    device = _device()
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def _params(traced: TracedModel) -> int:
    return sum(p.numel() for p in traced.gm.parameters())


def _block_action(traced: TracedModel, operation: str, settings: dict) -> ResidualBlockAction | None:
    kwargs = {k: v for k, v in settings.items() if k != "shrink_mode"}
    grow = operation == "duplicate"
    actions = ResidualBlockAction.generate_all_actions(
        traced, grow=grow, shrink=not grow, shrink_mode=settings.get("shrink_mode", "remove"), **kwargs,
    )
    return next((a for a in actions if a.params[0] == operation), None)


def _paired_row(checkpoint: TracedModel, label: str, operation: str, settings: dict | None,
                train_loader, val_loader, seeds) -> dict:
    base_acc = _accuracy(checkpoint.gm, val_loader)
    edited = copy.deepcopy(checkpoint)
    if settings is not None:
        action = _block_action(edited, operation, settings)
        if action is None:
            return {"setting": label, "available": False}
        action.execute(edited)
    now = _accuracy(edited.gm, val_loader)
    after_one, after_gen = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = copy.deepcopy(edited.gm)
        _train(model, train_loader, 1, GENERATION_LR)
        after_one.append(_accuracy(model, val_loader))
        _train(model, train_loader, GENERATION_EPOCHS - 1, GENERATION_LR)
        after_gen.append(_accuracy(model, val_loader))
    return {
        "setting": label,
        "available": True,
        "params_added": _params(edited) - _params(checkpoint),
        "checkpoint_val": base_acc,
        "val_change_now": now - base_acc,
        "val_after_1_epoch": statistics.mean(after_one),
        "val_after_generation": statistics.mean(after_gen),
        "val_after_generation_by_seed": after_gen,
    }


def run_part_a(scale: float, seeds) -> None:
    path = OUT_DIR / "paired" / "paired_results.json"
    if path.is_file():
        print(f"SKIP Part A: results exist at {path} (delete the file to rerun)", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed_all(0)
    data = get_datasets(DATASET)
    train_loader, val_loader, _ = data.loaders(BATCH_SIZE)
    train_loader = with_cutout(train_loader)

    # Checkpoint 1: the Experiment 014 starter after a short training run.
    plain = TracedModel.create(ResidualStarter(3, 10), INPUT_SHAPE)
    _train(plain.gm, train_loader, CHECKPOINT_EPOCHS, CHECKPOINT_LR)
    plain.invalidate()
    # Checkpoint 2: the same model after AddResConvLayer wired the stem into the block.
    side = copy.deepcopy(plain)
    next(a for a in AddResConvLayer.generate_all_actions(side) if a.params[:2] == ["stem", "block_conv"]).execute(side)
    _train(side.gm, train_loader, 5, CHECKPOINT_LR)
    side.invalidate()
    # Checkpoint 3: two identical blocks (after-ReLU duplicate, trained) for the shrink comparison.
    two = copy.deepcopy(plain)
    _block_action(two, "duplicate", dict(scale=1.0, after_activation=True, allow_side_inputs=False)).execute(two)
    _train(two.gm, train_loader, GENERATION_EPOCHS, CHECKPOINT_LR)
    two.invalidate()
    print(f"Part A checkpoints ready in {time.time() - t0:.0f}s", flush=True)

    steps = ladder(scale)
    grow_settings = [("no change", None)] + [(name, s) for name, s in steps[1:5]]
    scale_sweep = [(f"scale {value:g}", {**steps[3][1], "scale": value}) for value in (1.0, 0.5, 0.3, 0.1, 0.0)]
    results = {"scale": scale, "seeds": list(seeds), "tables": {}}
    for title, checkpoint, rows in (
        ("Grow ladder, plain checkpoint", plain, grow_settings),
        ("Grow ladder, checkpoint with AddResConv side input", side, grow_settings),
        ("Scale sweep (side + after_relu), plain checkpoint", plain, [("no change", None)] + scale_sweep),
    ):
        table = []
        for label, settings in rows:
            row = _paired_row(checkpoint, label, "duplicate", settings, train_loader, val_loader, seeds)
            table.append(row)
            print(f"  [{title}] {label}: {row}", flush=True)
        results["tables"][title] = table
    shrink = []
    for label, op, settings in (
        ("no change", "remove", None),
        ("remove", "remove", dict(scale=1.0, after_activation=True, allow_side_inputs=True, shrink_mode="remove")),
        ("merge", "merge", dict(scale=1.0, after_activation=True, allow_side_inputs=True, shrink_mode="merge")),
    ):
        row = _paired_row(two, label, op, settings, train_loader, val_loader, seeds)
        shrink.append(row)
        print(f"  [Shrink, two-block checkpoint] {label}: {row}", flush=True)
    results["tables"]["Shrink, two-block checkpoint"] = shrink
    results["duration_sec"] = time.time() - t0
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Part A done in {results['duration_sec'] / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Part B: growing runs
# ---------------------------------------------------------------------------

def _configure(config: RunningConfig, settings: dict | None) -> None:
    enabled = settings is not None
    config.ACTIONS_ENABLE_RESIDUAL_BLOCK_GROW = enabled
    config.ACTIONS_ENABLE_RESIDUAL_BLOCK_SHRINK = enabled
    if enabled:
        config.RESIDUAL_BLOCK_DUPLICATE_SCALE = settings["scale"]
        config.RESIDUAL_BLOCK_DUPLICATE_AFTER_ACTIVATION = settings["after_activation"]
        config.RESIDUAL_BLOCK_ALLOW_SIDE_INPUTS = settings["allow_side_inputs"]
        config.RESIDUAL_BLOCK_SHRINK_MODE = settings["shrink_mode"]


def _train_one(data, name: str, settings: dict | None, seed: int, scale: float) -> None:
    run_dir = OUT_DIR / "growing" / arm_folder(name, scale) / f"seed_{seed}"
    if not _claim_run_dir(run_dir):
        return
    try:
        seed_all(seed)
        model = ResidualStarter(3, 10)
        train_loader, val_loader, clean_loader = data.loaders(BATCH_SIZE)
        train_loader = with_cutout(train_loader)
        test_loader = data.test_loader(BATCH_SIZE)
        sim_train, sim_val = sample_loaders(clean_loader, val_loader, SIMULATION_SET_SIZE, seed=seed)
        device = _device()
        board = ExperimentBoard(
            run_dir / "board",
            experiment_id=f"{EXPERIMENT_ID}_{arm_folder(name, scale)}_seed{seed}",
            experiment_name=f"Experiment 015 duplicate action design: {name}",
            dataset=DATASET.upper(),
            device=str(device),
        )
        config = RunningConfig(
            generations=GENERATIONS,
            epochs=EPOCHS,
            lr_scheduler=CosineDecayScheduler(total_epochs=GENERATIONS * EPOCHS),
            stopper=AccuracyStopper(
                target_accuracy=TARGET_VALIDATION_ACCURACY, metric_name="validation accuracy", metric_key="val_acc",
            ),
            simulation_alg=montecarlo_alg,
            simulation_scheduler=AlwaysSimulationScheduler(SIMULATION_SECONDS, SIMULATION_EPOCHS),
            simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
            simulation_set_size=SIMULATION_SET_SIZE,
            criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
            enable_experiment_board=True,
            experiment_board=board,
            device=device,
            momentum=0.9,
            weight_decay=0.0001,
            max_grad_norm=1.0,
            seq_conv_insert_before_global_pool=True,
        )
        _configure(config, settings)
        print(f"EXP014 arm={name} seed={seed} settings={settings} device={device}", flush=True)
        _, summary = train_generations(
            model, train_loader, val_loader, config,
            sim_train_loader=sim_train, sim_val_loader=sim_val, test_loader=test_loader,
        )
        history = {
            "experiment": EXPERIMENT_ID,
            "arm": name,
            "seed": seed,
            "settings": settings,
            **{key: list(value) for key, value in summary.items()},
        }
        torch.save(history, run_dir / "history.pt")
        (run_dir / "metadata.json").write_text(
            json.dumps({k: v for k, v in history.items() if not isinstance(v, list)}, indent=2), encoding="utf-8",
        )
        print(f"DONE {name}/seed_{seed}", flush=True)
    except Exception as exc:
        (run_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def run_part_b(scale: float, seeds, arms) -> None:
    data = get_datasets(DATASET)
    steps = [step for step in ladder(scale) if not arms or step[0] in arms]
    for seed in seeds:
        for name, settings in steps:
            _train_one(data, name, settings, seed, scale)
            write_summary(scale)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _block_op(action_text: str) -> str | None:
    if "Residual Block" not in action_text:
        return None
    for op in ("duplicate", "merge", "remove"):
        if f"'{op}'" in action_text:
            return op
    return None


def _search_stats(board_dir: Path) -> dict:
    """Per-run residual-block statistics from the saved Monte Carlo candidates."""
    searches = offered = 0
    vs_no_change, ranks = [], []
    chosen = {"duplicate": 0, "merge": 0, "remove": 0}
    for path in sorted((board_dir / "simulations").glob("simulation_gen_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        candidates = data.get("candidates") or []
        if not candidates:
            continue
        searches += 1
        scores = sorted((c["score"] for c in candidates), reverse=True)
        dropout = [c["score"] for c in candidates if "Dropout" in c.get("name", "")]
        baseline = statistics.median(dropout) if dropout else None
        duplicates = [c for c in candidates if _block_op(c.get("action", "")) == "duplicate"]
        if duplicates:
            offered += 1
        for c in duplicates:
            ranks.append(scores.index(c["score"]) / max(1, len(scores) - 1))
            if baseline is not None:
                vs_no_change.append(c["score"] - baseline)
        for c in candidates:
            op = _block_op(c.get("action", ""))
            if op and c.get("chosen"):
                chosen[op] += 1
    return {
        "searches": searches,
        "searches_offering_duplicate": offered,
        "duplicate_score_vs_no_change": statistics.mean(vs_no_change) if vs_no_change else None,
        "duplicate_rank": statistics.mean(ranks) if ranks else None,
        "chosen": chosen,
    }


def _run_stats(run_dir: Path) -> dict | None:
    history_path = run_dir / "history.pt"
    if not history_path.is_file():
        return None
    h = torch.load(history_path, map_location="cpu", weights_only=False)
    executed = {"duplicate": 0, "merge": 0, "remove": 0}
    for name, op in zip(h.get("action_name", []), h.get("action_from", [])):
        if name == "ResidualBlockAction" and op in executed:
            executed[op] += 1
    last = lambda key: float(h[key][-1]) if h.get(key) else None
    return {
        "train": last("train_acc"),
        "val": last("val_acc"),
        "test": last("test_acc"),
        "params": int(h["param_count"][-1]) if h.get("param_count") else None,
        "epochs": len(h.get("train_acc", [])),
        "executed": executed,
        **_search_stats(run_dir / "board"),
    }


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _signed_pp(value) -> str:
    return "—" if value is None else f"{100 * value:+.2f} pp"


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def write_summary(scale: float) -> str:
    lines = [f"# Experiment 015 summary", "", f"Scale used by the +scale and +merge steps: {scale:g}.", ""]
    paired = OUT_DIR / "paired" / "paired_results.json"
    if paired.is_file():
        data = json.loads(paired.read_text(encoding="utf-8"))
        lines += ["## Part A: same checkpoint, one generation of training", "",
                  "Changes are against the checkpoint's validation accuracy. `vs no change` compares the "
                  "after-generation accuracy with the no-change row of the same table; higher is better.", ""]
        for title, rows in data["tables"].items():
            lines += [f"### {title}", "",
                      "| Setting | Available | +Params | Val change right after | Val after 1 epoch | Val after generation | vs no change |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            reference = next((r for r in rows if r["setting"] == "no change" and r["available"]), None)
            for r in rows:
                if not r["available"]:
                    lines.append(f"| {r['setting']} | no candidate | — | — | — | — | — |")
                    continue
                delta = None if reference is None else r["val_after_generation"] - reference["val_after_generation"]
                lines.append(
                    f"| {r['setting']} | yes | {r['params_added']:+d} | {_signed_pp(r['val_change_now'])} | "
                    f"{_pct(r['val_after_1_epoch'])} | {_pct(r['val_after_generation'])} | {_signed_pp(delta)} |"
                )
            lines.append("")
    rows = []
    for name, _ in ladder(scale):
        arm_dir = OUT_DIR / "growing" / arm_folder(name, scale)
        stats = [s for s in (_run_stats(d) for d in sorted(arm_dir.glob("seed_*")) if "_" not in d.name[5:]) if s]
        rows.append((name, stats))
    if any(stats for _, stats in rows):
        lines += ["## Part B: growing runs (Experiment 014 CIFAR-10 protocol)", "",
                  "Each row adds one change to the row above. Final values are the last recorded epoch, "
                  "averaged over finished seeds. `Score vs no change` is the duplicate's simulation score minus "
                  "the median dropout-candidate score in the same search (dropout does not change predictions).", "",
                  "| Step | Seeds | Test | Test change vs previous | Val | Params | Searches offering duplicate | "
                  "Score vs no change | Rank (0 best) | Chosen dup/shrink | Executed dup/shrink |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        previous = None
        for name, stats in rows:
            if not stats:
                lines.append(f"| {name} | 0 | — | — | — | — | — | — | — | — | — |")
                continue
            test = _mean(s["test"] for s in stats)
            delta = None if previous is None or test is None else test - previous
            previous = test if test is not None else previous
            searches = sum(s["searches"] for s in stats)
            offered = sum(s["searches_offering_duplicate"] for s in stats)
            score = _mean(s["duplicate_score_vs_no_change"] for s in stats)
            rank = _mean(s["duplicate_rank"] for s in stats)
            chosen = (sum(s["chosen"]["duplicate"] for s in stats),
                      sum(s["chosen"]["remove"] + s["chosen"]["merge"] for s in stats))
            executed = (sum(s["executed"]["duplicate"] for s in stats),
                        sum(s["executed"]["remove"] + s["executed"]["merge"] for s in stats))
            params = _mean(s["params"] for s in stats)
            lines.append(
                f"| {name} | {len(stats)} | {_pct(test)} | {_signed_pp(delta)} | {_pct(_mean(s['val'] for s in stats))} | "
                f"{'—' if params is None else f'{params:,.0f}'} | {offered}/{searches} | "
                f"{'—' if score is None else f'{score:+.3f}'} | {'—' if rank is None else f'{rank:.2f}'} | "
                f"{chosen[0]}/{chosen[1]} | {executed[0]}/{executed[1]} |"
            )
        lines += ["", "Per seed test accuracy:", ""]
        for name, stats in rows:
            if stats:
                lines.append(f"- {name}: " + ", ".join(_pct(s["test"]) for s in stats))
    text = "\n".join(lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("a", "b", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--scale", default="auto",
                        help="scale for the +scale and +merge steps: a number in [0, 1], or auto "
                             "(best of the Part A scale sweep; 0.3 if Part A has not run)")
    parser.add_argument("--arms", nargs="+", choices=[name for name, _ in ladder(DEFAULT_SCALE)],
                        help="Part B steps to run (default: all)")
    parser.add_argument("--summary", action="store_true", help="only print and write summary.md")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        scale, source = resolve_scale(args.scale)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.summary:
        parser.error("the compared design switches were removed after this experiment; only --summary runs")
        if args.part in ("a", "all"):
            run_part_a(scale, args.seeds)
            scale, source = resolve_scale(args.scale)
            write_summary(scale)
        if args.part in ("b", "all"):
            print(f"Part B scale for +scale and +merge: {scale:g} ({source})", flush=True)
            run_part_b(scale, args.seeds, args.arms)
    print(write_summary(scale), flush=True)


if __name__ == "__main__":
    main()
