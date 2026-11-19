"""Experiment 010: compress GPT-2 with GrowingNN architecture search.

The frozen GPT-2 teacher produces a reproducible next-token memory snapshot.
GrowingNN's existing trainer and Monte Carlo action search then train and score
the captured student against that snapshot, with a locally calibrated size term.

Run from the repository root after ``pipenv shell``::

    python ./experiments/train_llm_exp010_gpt2_compression.py --smoke
    python ./experiments/train_llm_exp010_gpt2_compression.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import growingnn.core.config as project_config
import growingnn.training.lr_scheduler_action as lr_config
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import ModuleResolver
from growingnn.simulation.score_functions.score_by_learning import score_acc
from growingnn.simulation.score_functions.score_efficiency import score_count_weights
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import EmptyStopper
from growingnn.training.trainer import train_generations
from growingnn.training.causal_lm import (
    CausalLMNextTokenAdapter,
    pack_token_inputs,
    teacher_token_agreement,
)

MODEL_ID = "openai-community/gpt2"
OUTPUT = ROOT / "experiments" / "output" / "train_llm" / "exp010_gpt2_compression"
PROMPT_TOPICS = (
    "a river crossing a dry valley", "a scientist testing a new material",
    "a small language model", "a city preparing for winter", "a robot learning a tune",
    "a telescope finding a distant planet", "a careful weather forecast",
    "an old library", "a team restoring a wetland", "a sequence of sensor readings",
    "a camera describing a busy street", "a student building a simple program",
    "an unexpected discovery in the ocean", "a train traveling through the mountains",
    "a garden adapting to less water", "a museum preserving a fragile object",
    "a new method for storing energy", "a community planning a public space",
    "a satellite measuring forest growth", "a musician composing with code",
    "a historian comparing two letters", "a medical device monitoring motion",
    "a chef adapting a family recipe", "a researcher studying animal migration",
    "a factory reducing material waste", "a school using a new learning tool",
    "a spacecraft preparing for a long journey", "a team analyzing economic trends",
    "a village responding to a power outage", "a map of changing coastlines",
)
PROMPT_FORMS = (
    "Explain {topic} in clear steps.",
    "Write a short, factual description of {topic}.",
    "What could happen next in a story about {topic}?",
    "Describe one challenge and one possible solution involving {topic}.",
    "Give a beginner-friendly example related to {topic}.",
)


class FixedOutputProjection(nn.Module):
    """Retain the tied vocabulary head while excluding unsafe output-width edits."""

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.weight = layer.weight
        self.bias = layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def _make_prompts(train_count: int, validation_count: int, seed: int) -> tuple[list[str], list[str]]:
    pool = [form.format(topic=topic) for form in PROMPT_FORMS for topic in PROMPT_TOPICS]
    random.Random(seed).shuffle(pool)
    required = train_count + validation_count
    if required > len(pool):
        raise ValueError(f"requested {required} prompts, but only {len(pool)} unique prompts are available")
    return pool[:train_count], pool[train_count:required]


def _snapshot(model, tokenizer, prompts: list[str], *, context: int, new_tokens: int,
              device: torch.device, batch_size: int, top_k: int = 8
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    records: list[dict] = []
    input_ids_rows: list[torch.Tensor] = []
    attention_rows: list[torch.Tensor] = []
    labels: list[int] = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=context).to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=new_tokens,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       return_dict_in_generate=True, output_scores=True)
        for row, prompt in enumerate(batch):
            prompt_ids = encoded.input_ids[row][encoded.attention_mask[row].bool()].tolist()
            continuation = generated.sequences[row, encoded.input_ids.shape[1]:].tolist()
            top_ids: list[list[int]] = []
            top_probs: list[list[float]] = []
            for step_scores in generated.scores:
                values, ids = step_scores[row].float().softmax(dim=-1).topk(top_k)
                top_ids.append(ids.cpu().tolist())
                top_probs.append([round(float(v), 7) for v in values.cpu().tolist()])
            full_ids = list(prompt_ids)
            for step, target in enumerate(continuation):
                prefix = full_ids[-context:]
                pad_count = context - len(prefix)
                input_ids_rows.append(torch.tensor([tokenizer.pad_token_id] * pad_count + prefix,
                                                   dtype=torch.long))
                attention_rows.append(torch.tensor([0] * pad_count + [1] * len(prefix),
                                                    dtype=torch.long))
                labels.append(int(target))
                full_ids.append(int(target))
            records.append({"prompt": prompt, "prompt_token_ids": prompt_ids,
                            "teacher_token_ids": continuation,
                            "top_k_token_ids": top_ids, "top_k_probabilities": top_probs})
    return (torch.stack(input_ids_rows), torch.stack(attention_rows),
            torch.tensor(labels, dtype=torch.long), records)


def _params(module: nn.Module) -> int:
    seen: set[int] = set()
    return sum(p.numel() for p in module.parameters() if not (id(p) in seen or seen.add(id(p))))


def _generation_agreement(model: nn.Module, records: list[dict], *, context: int,
                          pad_token_id: int, device: torch.device) -> dict[str, float | int]:
    model.eval()
    token_matches = token_total = exact_sequences = 0
    with torch.inference_mode():
        for record in records:
            sequence = list(record["prompt_token_ids"])
            expected = record["teacher_token_ids"]
            matches = 0
            for target in expected:
                prefix = sequence[-context:]
                pad = context - len(prefix)
                ids = [pad_token_id] * pad + prefix
                mask = [0] * pad + [1] * len(prefix)
                packed = torch.tensor([ids, mask], dtype=torch.float32, device=device).unsqueeze(0)
                prediction = int(model(packed).argmax(dim=-1).item())
                matches += prediction == int(target)
                token_matches += prediction == int(target)
                token_total += 1
                sequence.append(prediction)
            exact_sequences += int(matches == len(expected))
    return {"matched_tokens": token_matches, "total_tokens": token_total,
            "token_accuracy": token_matches / token_total if token_total else 0.0,
            "exact_sequences": exact_sequences, "total_sequences": len(records),
            "exact_sequence_fraction": exact_sequences / len(records) if records else 0.0}


def _benchmark_prompts(used_prompts: list[str], *, count: int, seed: int) -> list[str]:
    """Choose a repeatable prompt set that is separate from train/validation."""
    used = set(used_prompts)
    pool = [form.format(topic=topic) for form in PROMPT_FORMS for topic in PROMPT_TOPICS]
    pool = [prompt for prompt in pool if prompt not in used]
    random.Random(seed).shuffle(pool)
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} unused prompts are available for a {count}-prompt benchmark")
    return pool[:count]


def _timed_greedy_prompts(model: nn.Module, tokenizer, prompts: list[str], *, context: int,
                          new_tokens: int, device: torch.device) -> dict:
    """Generate greedy continuations with a sliding window only at the model limit."""
    model.eval()
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False)[-context:]
                  for prompt in prompts]

    def next_token(ids: list[int]) -> int:
        prefix = ids[-context:]
        # Use only real tokens; padding every example to the training context
        # silently creates a 32-token sliding window and changes longer outputs.
        row_ids = torch.tensor([prefix], dtype=torch.long)
        row_mask = torch.ones_like(row_ids)
        packed = pack_token_inputs(row_ids, row_mask).to(device)
        return int(model(packed).argmax(dim=-1).item())

    # Warm up each model before measuring the same prompt workload.
    with torch.inference_mode():
        next_token(prompt_ids[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generations = []
    started = time.perf_counter()
    with torch.inference_mode():
        for prompt, initial_ids in zip(prompts, prompt_ids):
            sequence = list(initial_ids)
            generated = []
            for _ in range(new_tokens):
                token = next_token(sequence)
                generated.append(token)
                sequence.append(token)
                if token == tokenizer.eos_token_id:
                    break
            generations.append({"prompt": prompt, "prompt_token_ids": initial_ids,
                                "generated_token_ids": generated,
                                "generated_text": tokenizer.decode(generated)})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {"elapsed_seconds": elapsed, "prompt_count": len(prompts),
            "seconds_per_prompt": elapsed / max(1, len(prompts)),
            "new_tokens_requested": new_tokens, "outputs": generations}


def _hf_greedy_prompts(model: nn.Module, tokenizer, prompts: list[str], *, context: int,
                       new_tokens: int, device: torch.device) -> list[dict]:
    """Collect reference continuations through Hugging Face ``generate``."""
    model.eval()
    records = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=context).to(device)
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            continuation = generated[0, encoded.input_ids.shape[1]:].tolist()
            records.append({"prompt": prompt, "prompt_token_ids": encoded.input_ids[0].tolist(),
                            "generated_token_ids": continuation,
                            "generated_text": tokenizer.decode(continuation)})
    return records


def _benchmark_output_agreement(before: dict, after: dict) -> dict[str, float | int]:
    matched = total = exact = 0
    for reference, result in zip(before["outputs"], after["outputs"]):
        expected = reference["generated_token_ids"]
        actual = result["generated_token_ids"]
        shared = min(len(expected), len(actual))
        matched += sum(left == right for left, right in zip(expected[:shared], actual[:shared]))
        total += max(len(expected), len(actual))
        exact += int(expected == actual)
    return {"matched_tokens": matched, "total_tokens": total,
            "token_accuracy": matched / total if total else 0.0,
            "exact_sequences": exact, "total_sequences": len(before["outputs"]),
            "exact_sequence_fraction": exact / len(before["outputs"]) if before["outputs"] else 0.0}


def _render_markdown_report(report: dict) -> str:
    """Render a readable, reproducible summary from the machine-readable report."""
    reduction = report["parameter_reduction_fraction"] * 100.0
    agreement = report["heldout_greedy_generation_agreement"]
    benchmark = report.get("same_prompt_generation_benchmark", {})
    before = benchmark.get("before_training", {})
    after = benchmark.get("after_training", {})
    benchmark_agreement = benchmark.get("output_agreement", {})
    history = report.get("history", {})
    epochs_per_generation = max(1, int(report.get("epochs_per_generation", 1)))
    lines = [
        f"# Experiment 010 run report: {report['model']}", "",
        f"- **Status:** {report['status']}",
        f"- **Run:** {report.get('planned_generations', '?')} generations × "
        f"{epochs_per_generation} epochs ({len(history.get('train_loss', []))} recorded epochs)",
        f"- **Device:** {report['device']}",
        f"- **Training and search time:** {report['training_seconds']:.1f} seconds "
        f"({report['training_seconds'] / 60:.1f} minutes)",
        f"- **Parameters:** {report['initial_parameters']:,} → {report['final_parameters']:,} "
        f"({reduction:.2f}% reduction)",
        f"- **Training teacher-token agreement:** {report['training_teacher_token_accuracy']:.1%}",
        f"- **Held-out teacher-token agreement:** {report['heldout_teacher_token_accuracy']:.1%} "
        f"(loss {report['heldout_teacher_token_loss']:.4f})",
        f"- **Held-out greedy generation agreement:** {agreement['matched_tokens']}/"
        f"{agreement['total_tokens']} tokens ({agreement['token_accuracy']:.1%}); "
        f"{agreement['exact_sequences']}/{agreement['total_sequences']} exact prompts",
    ]
    capture_agreement = report.get("initial_capture_teacher_generation_agreement")
    if capture_agreement:
        lines.append(
            f"- **Initial FX vs Hugging Face generation check:** "
            f"{capture_agreement['matched_tokens']}/{capture_agreement['total_tokens']} tokens "
            f"({capture_agreement['token_accuracy']:.1%})"
        )
    chart = report.get("training_history_chart")
    if chart:
        lines.extend(["", f"![Training and held-out loss, agreement, and parameter count over "
                      f"the {len(history.get('train_loss', []))} recorded epochs]({chart})", ""])
    lines.extend(["", "## Before and after prompt timing", ""])
    if before and after:
        lines.extend([
            f"The same {before['prompt_count']} held-aside prompts took "
            f"{before['elapsed_seconds']:.3f}s before training and "
            f"{after['elapsed_seconds']:.3f}s after training "
            f"({benchmark_agreement.get('token_accuracy', 0.0):.1%} generated-token agreement; "
            f"{benchmark_agreement.get('exact_sequences', 0)}/"
            f"{benchmark_agreement.get('total_sequences', 0)} exact continuations).",
            "", "| Prompt | Before | After |", "| --- | --- | --- |",
        ])
        after_by_prompt = {item["prompt"]: item["generated_text"]
                           for item in after.get("outputs", [])}
        for item in before.get("outputs", []):
            prompt = item["prompt"].replace("|", "\\|")
            old_text = item["generated_text"].replace("|", "\\|").replace("\n", " ")
            new_text = after_by_prompt.get(item["prompt"], "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {prompt} | {old_text} | {new_text} |")
    extended = report.get("extended_generation_examples", {})
    extended_before = extended.get("before_training", {})
    extended_after = extended.get("after_training", {})
    if extended_before and extended_after:
        long_agreement = extended.get("output_agreement", {})
        lines.extend([
            "", f"## Readable generation examples ({extended.get('new_tokens_requested', '?')} tokens)", "",
            f"These examples use the full GPT-2 context limit ({extended.get('context_limit', '?')} tokens), "
            "rather than truncating the prompt to the short training window.",
            f"The longer generation workload took {extended_before['elapsed_seconds']:.3f}s "
            f"before training and {extended_after['elapsed_seconds']:.3f}s after training. "
            f"Outputs on the same prompts agreed for "
            f"{long_agreement.get('matched_tokens', 0)}/{long_agreement.get('total_tokens', 0)} "
            f"tokens ({long_agreement.get('token_accuracy', 0.0):.1%}); "
            f"{long_agreement.get('exact_sequences', 0)}/"
            f"{long_agreement.get('total_sequences', 0)} continuations matched exactly.",
            "",
        ])
        extended_after_by_prompt = {item["prompt"]: item["generated_text"]
                                    for item in extended_after.get("outputs", [])}
        for index, item in enumerate(extended_before.get("outputs", [])[:5], start=1):
            prompt = item["prompt"].replace("|", "\\|").replace("\n", " ")
            old_text = item["generated_text"].replace("|", "\\|").replace("\n", " ")
            new_text = extended_after_by_prompt.get(item["prompt"], "")
            new_text = new_text.replace("|", "\\|").replace("\n", " ")
            lines.extend([f"### Example {index}", "", f"**Prompt:** {prompt}", "",
                          f"**Original-size model:** {old_text}", "",
                          f"**Compressed model:** {new_text}", ""])
    progression = report.get("generation_examples", {})
    if progression.get("prompts") and progression.get("generations"):
        original_by_prompt = {item["prompt"]: item["generated_text"]
                              for item in progression.get("original_outputs", [])}
        lines.extend(["", "## Output after each search generation", "",
                      "Each row shows the student after that generation's search step; "
                      "generation 9 has no following action.", ""])
        for prompt in progression["prompts"]:
            lines.extend([f"### {prompt}", "",
                          f"**Original GPT-2:** {original_by_prompt.get(prompt, '')}", "",
                          "| Student generation | Action applied | Generated continuation |",
                          "| ---: | --- | --- |"])
            for generation in progression["generations"]:
                item = next((output for output in generation["outputs"]
                             if output["prompt"] == prompt), {})
                action = generation.get("action") or "none (final generation)"
                continuation = item.get("generated_text", "").replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {generation['generation']} | {action} | {continuation} |")
            lines.append("")
    lines.extend(["", "## Per-generation training history", "",
                  "| Generation | Train loss (first → last) | Held-out loss (first → last) | "
                  "Held-out agreement | Parameters |", "| ---: | ---: | ---: | ---: | ---: |"])
    generations = history.get("generation", [])
    for index, generation in enumerate(generations):
        start = index * epochs_per_generation
        end = min(start + epochs_per_generation, len(history.get("train_loss", []))) - 1
        if end < start:
            continue
        params = history.get("param_count", [])
        param_count = params[end] if end < len(params) else report["final_parameters"]
        lines.append(
            f"| {generation + 1} | {history['train_loss'][start]:.4f} → "
            f"{history['train_loss'][end]:.4f} | {history['val_loss'][start]:.4f} → "
            f"{history['val_loss'][end]:.4f} | {history['val_acc'][end]:.1%} | {param_count:,} |"
        )
    lines.extend(["", "## Selected actions", "", "| After generation | Action | Target | Setting |",
                  "| ---: | --- | --- | ---: |"])
    for generation, name, target, setting in zip(
        history.get("action_generation", []), history.get("action_name", []),
        history.get("action_from", []), history.get("action_to", []),
    ):
        lines.append(f"| {generation + 1} | {name} | `{target}` | {setting} |")
    lines.extend(["", "## Candidate diagnostics", "",
                  f"- Eligible shrink candidates: {report['eligible_shrink_actions']}",
                  f"- Offered to MCTS: {report['evaluated_shrink_actions']}"])
    reason_counts = Counter(item["reason"] for item in report.get("rejected_candidates", []))
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- Rejected/deferred ({count}): {reason}")
    lines.extend(["", "## Checkpoints and source data", "",
                  "This report is generated from `report.json`. Checkpoints and the teacher memory "
                  "snapshot are stored beside it.", ""])
    return "\n".join(lines)


def _write_training_history_chart(report: dict, destination: Path) -> bool:
    """Save a three-panel plot of losses, held-out agreement, and model size."""
    history = report.get("history", {})
    losses = history.get("train_loss", [])
    if not losses:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = list(range(1, len(losses) + 1))
    epochs_per_generation = max(1, int(report.get("epochs_per_generation", 1)))
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(epochs, losses, label="Training loss", color="#2563eb")
    axes[0].plot(epochs, history.get("val_loss", [])[:len(epochs)],
                 label="Held-out loss", color="#dc2626")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(epochs, history.get("val_acc", [])[:len(epochs)],
                 label="Held-out teacher-token agreement", color="#15803d")
    axes[1].set_ylabel("Token agreement")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    parameter_history = history.get("param_count", [])[:len(epochs)]
    if parameter_history:
        axes[2].plot(epochs[:len(parameter_history)],
                     [value / 1_000_000 for value in parameter_history],
                     label="Parameters", color="#7c3aed")
    axes[2].set_ylabel("Parameters (millions)")
    axes[2].set_xlabel("Training epoch")
    axes[2].grid(True, alpha=0.25)
    for axis in axes:
        for boundary in range(epochs_per_generation, len(epochs), epochs_per_generation):
            axis.axvline(boundary + 0.5, color="#64748b", linestyle="--", alpha=0.28)
    figure.suptitle("GPT-2 compression: training history")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return True


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="small short feasibility run")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--generations", type=int, default=None,
                   help="override generation count (default: 10 for larger runs, 2 for smoke)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs per generation (default: 10 for larger runs, 1 for smoke)")
    p.add_argument("--local-files-only", action="store_true")
    # Keep the ordinary one-command run on CPU. Opt into GPU later with
    # ``--device cuda``; ``auto`` remains available for memory-aware selection.
    p.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"),
                   help="compute device (default: cpu; use --device cuda to opt into GPU)")
    return p.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes >= 12 * 1024**3:
            return torch.device("cuda")
        print(f"CUDA has only {free_bytes / 1024**3:.1f} GiB free for MCTS graph copies; "
              "selecting CPU to avoid CUDA out-of-memory.",
              flush=True)
    return torch.device("cpu")


def main() -> None:
    args = _parse()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = 17
    random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(args.device)
    limits = {"context": 8, "new": 2, "train_prompts": 4, "validation_prompts": 2, "batch": 1,
              "epochs": 1, "generations": 2, "sim_seconds": 4.0, "lr": 1e-5}
    if not args.smoke:
        limits = {"context": 32, "new": 4, "train_prompts": 24, "validation_prompts": 8, "batch": 4,
                  "epochs": 10, "generations": 10, "sim_seconds": 3.0, "lr": 2e-5}
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        limits["epochs"] = args.epochs
    if args.generations is not None:
        if args.generations <= 0:
            raise ValueError("--generations must be positive")
        limits["generations"] = args.generations
    if args.output_dir is None:
        if args.smoke and args.generations is None and args.epochs is None:
            args.output_dir = OUTPUT / "smoke"
        else:
            args.output_dir = OUTPUT / f"g{limits['generations']}_e{limits['epochs']}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run plan: {limits['generations']} generations x {limits['epochs']} epochs = "
          f"{limits['generations'] * limits['epochs']} training epochs; "
          f"outputs will be saved to {args.output_dir}", flush=True)
    print(f"Loading unquantized {args.model} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                   local_files_only=args.local_files_only).to(device).eval()
    initial_params = _params(teacher)
    train_prompts, validation_prompts = _make_prompts(
        limits["train_prompts"], limits["validation_prompts"], seed
    )
    torch.save({"model_id": args.model, "state_dict": teacher.state_dict(),
                "parameter_count": initial_params, "seed": seed},
               args.output_dir / "teacher_checkpoint.pt")
    teacher_checkpoint_path = args.output_dir / "teacher_checkpoint.pt"
    print("Generating frozen-teacher memory snapshot ...", flush=True)
    train_ids, train_masks, train_targets, train_snapshot = _snapshot(
        teacher, tokenizer, train_prompts, context=limits["context"], new_tokens=limits["new"],
        device=device, batch_size=limits["batch"]
    )
    val_ids, val_masks, val_targets, val_snapshot = _snapshot(
        teacher, tokenizer, validation_prompts, context=limits["context"], new_tokens=limits["new"],
        device=device, batch_size=limits["batch"]
    )
    benchmark_prompts = _benchmark_prompts(
        train_prompts + validation_prompts, count=10, seed=seed + 1000
    )
    generation_prompts = benchmark_prompts[:3]
    generation_context = int(getattr(teacher.config, "max_position_embeddings",
                                     getattr(teacher.config, "n_positions", 1024)))
    print("Saving original GPT-2 answers for three generation-tracking prompts ...", flush=True)
    original_generation_outputs = _hf_greedy_prompts(
        teacher, tokenizer, generation_prompts, context=generation_context,
        new_tokens=24, device=device
    )
    x_train = pack_token_inputs(train_ids, train_masks)
    x_val = pack_token_inputs(val_ids, val_masks)
    train_ds = TensorDataset(x_train, train_targets)
    val_ds = TensorDataset(x_val, val_targets)
    train_loader = DataLoader(train_ds, batch_size=limits["batch"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=limits["batch"])
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # The wrapper casts trainer inputs back to token IDs. HFTracer preserves GPT-2
    # transformer operations, and editable capture normalizes Conv1D projections.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 local_files_only=args.local_files_only).to(device).eval()
    teacher_state = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(teacher_state["state_dict"], strict=True)
    del teacher_state
    wrapped = CausalLMNextTokenAdapter(model)
    sample = x_train[:1].to(device)
    traced = TracedModel.create(wrapped, tuple(sample.shape), capture="hf_symbolic_editable",
                                example_args=(sample,))
    # The vocabulary projection is the model output interface. Shrinking it would
    # change the tokenizer's class space and allocate a huge dense rescale matrix.
    for path, module in list(traced.gm.named_modules()):
        if path and isinstance(module, nn.Linear) and module.out_features == len(tokenizer):
            traced.gm.set_submodule(path, FixedOutputProjection(module))
    traced.shapes()
    captured_params = traced.param_count()
    print("Timing the initial student on 10 held-aside prompts ...", flush=True)
    benchmark_before = _timed_greedy_prompts(
        traced.gm, tokenizer, benchmark_prompts, context=limits["context"],
        new_tokens=limits["new"], device=device
    )
    print(f"Initial benchmark: {benchmark_before['elapsed_seconds']:.2f}s total; "
          f"{benchmark_before['seconds_per_prompt']:.2f}s/prompt", flush=True)
    print("Checking the captured initial student on the three full-context examples ...", flush=True)
    original_size_examples = _timed_greedy_prompts(
        traced.gm, tokenizer, generation_prompts, context=generation_context,
        new_tokens=24, device=device
    )
    initial_capture_agreement = _benchmark_output_agreement(
        {"outputs": original_generation_outputs}, original_size_examples
    )
    print(f"Initial captured-vs-Hugging-Face generation agreement: "
          f"{initial_capture_agreement['matched_tokens']}/"
          f"{initial_capture_agreement['total_tokens']} tokens", flush=True)

    score = SimulationScore(weight_acc=0.1, weight_countW=1.0)
    # Keep the project's score implementation and algorithm; calibrate only the
    # global size scale locally so the baseline size term is 0.5.
    old_size_weight = project_config.WEIGHT_COUNT_WEIGHT
    project_config.WEIGHT_COUNT_WEIGHT = 1.0 / max(1, captured_params)
    original_score_functions = score._SCORE_FUNCTIONS
    score_evaluations: list[dict] = []
    def accuracy_for_score(candidate, cfg):
        result = score_acc(candidate, cfg)
        cfg._exp010_candidate_accuracy = result
        return result
    def calibrated_size_score(candidate, cfg):
        accuracy = getattr(cfg, "_exp010_candidate_accuracy", 0.0)
        size_score = score_count_weights(candidate, cfg)
        score_evaluations.append({
            "parameters": _params(candidate), "accuracy_score": float(accuracy),
            "size_score": float(size_score),
            "weighted_accuracy": 0.1 * float(accuracy),
            "weighted_size": float(size_score),
            "composite": (0.1 * float(accuracy) + float(size_score)) / 1.1,
        })
        return float(size_score)
    score._SCORE_FUNCTIONS = {**original_score_functions,
                              "weight_acc": accuracy_for_score,
                              "weight_countW": calibrated_size_score}

    cfg = RunningConfig(generations=limits["generations"], epochs=limits["epochs"],
                        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=limits["lr"]),
                        stopper=EmptyStopper(), simulation_alg=montecarlo,
                        simulation_scheduler=AlwaysSimulationScheduler(limits["sim_seconds"], 1),
                        simulation_score=score,
                        simulation_set_size=min(2 if args.smoke else 8, len(train_ds)),
                        criterion=nn.CrossEntropyLoss(), quiet=False, print_every=1, device=device)
    cfg.set_simulation_loaders(train_loader, val_loader)
    baseline_acc = teacher_token_agreement(traced.gm, val_loader, device)
    cfg.update_grow_actions(False)
    cfg.ACTIONS_ENABLE_DEL_LAYER = False
    cfg.ACTIONS_ENABLE_DEL_NEURONS_01 = False
    cfg.ACTIONS_ENABLE_DEL_NEURONS_05 = False
    cfg.ACTIONS_ENABLE_DEL_NEURONS_09 = True
    previous_min_lr = lr_config.MIN_LEARNING_RATE
    try:
        # The project-wide scheduler enforces a 1e-3 floor, which is destructive
        # for pretrained GPT-2. Lower it only within this experiment's run.
        lr_config.MIN_LEARNING_RATE = 1e-6
        print(f"Captured {len(list(traced.gm.graph.nodes))} FX nodes; discovering shrink candidates ...", flush=True)
        from growingnn.actions.delete_neurons import DelNeurons
        from growingnn.actions.utils.layer_resize import linear_output_resize_rejection_reason
        actions = DelNeurons.generate_all_actions(traced, ratio=0.9)
        eligible_count = len(actions)
        eligible_candidate_targets = [str(action.params[0]) for action in actions]
        rejected_candidates = []
        for path, module in traced.gm.named_modules():
            if not path or not isinstance(module, nn.Linear):
                continue
            new_width = max(1, int(module.out_features * 0.9))
            reason = linear_output_resize_rejection_reason(traced.gm, path, new_width)
            if reason is not None:
                rejected_candidates.append({"target": path, "reason": reason})
        for path, module in traced.gm.named_modules():
            if path and isinstance(module, FixedOutputProjection):
                rejected_candidates.append({"target": path,
                                            "reason": "vocabulary output width is pinned to tokenizer classes"})
        # MCTS currently retains a GraphModule copy for each expanded node. Bound
        # the number of copies per round, and sample candidates reproducibly so
        # target order does not decide which operations are even considered.
        candidate_budget = min(1 if args.smoke else 3, len(actions))
        if len(actions) > candidate_budget:
            chosen = random.Random(seed + 100).sample(actions, candidate_budget)
            chosen_targets = {action.params[0] for action in chosen}
            for action in actions:
                if action.params[0] in chosen_targets:
                    continue
                module = ModuleResolver.get_layer_module(action.params[0], traced.gm)
                traced.gm.set_submodule(action.params[0], FixedOutputProjection(module))
                rejected_candidates.append({
                    "target": action.params[0],
                    "reason": "deferred this round to respect MCTS graph-copy memory budget",
                })
            actions = DelNeurons.generate_all_actions(traced, ratio=0.9)
        print(f"Eligible 10% neuron-deletion actions: {eligible_count} total; "
              f"{len(actions)} offered to the existing search (candidate memory budget="
              f"{candidate_budget})", flush=True)
        torch.save({"state_dict": traced.gm.state_dict(), "graph_code": traced.gm.code,
                    "parameter_count": captured_params}, args.output_dir / "student_initial_checkpoint.pt")
        best = {"score": float("-inf"), "generation": None, "parameters": None,
                "heldout_teacher_token_accuracy": None}
        generation_progress = []
        generation_outputs_path = args.output_dir / "generation_outputs.json"
        generation_output_payload = {
            "status": "in_progress", "prompts": generation_prompts,
            "original_outputs": original_generation_outputs, "generations": generation_progress,
        }
        generation_outputs_path.write_text(json.dumps(generation_output_payload, indent=2),
                                           encoding="utf-8")
        progress_history = {key: [] for key in
                            ("generation", "train_loss", "train_acc", "val_loss", "val_acc", "lr")}
        def save_best_student(current_model, generation, generation_history):
            for key in progress_history:
                if key == "generation":
                    progress_history[key].extend([generation] * len(generation_history["train_loss"]))
                else:
                    progress_history[key].extend(generation_history[key])
            progress = {
                "status": "training_in_progress", "completed_generation_training": generation + 1,
                "planned_generations": limits["generations"], "planned_epochs_per_generation": limits["epochs"],
                "parameters": _params(current_model), "history": progress_history,
            }
            (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
            torch.save({"state_dict": current_model.state_dict(), "graph_code": current_model.code,
                        "parameter_count": _params(current_model), "generation": generation,
                        "history": progress_history}, args.output_dir / "student_progress_checkpoint.pt")
            print(f"Completed generation {generation + 1}/{limits['generations']} training: "
                  f"train_loss={generation_history['train_loss'][-1]:.4f}, "
                  f"heldout_loss={generation_history['val_loss'][-1]:.4f}, "
                  f"heldout_agreement={generation_history['val_acc'][-1]:.3f}, "
                  f"parameters={_params(current_model):,}", flush=True)
            val_accuracy = float(generation_history["val_acc"][-1])
            params = _params(current_model)
            size_term = 1.0 / (params / max(1, captured_params) + 1.0)
            composite = (0.1 * val_accuracy + size_term) / 1.1
            if composite <= best["score"]:
                return
            best.update(score=composite, generation=generation, parameters=params,
                        heldout_teacher_token_accuracy=val_accuracy)
            torch.save({"state_dict": current_model.state_dict(), "graph_code": current_model.code,
                        "parameter_count": params, "generation": generation,
                        "heldout_teacher_token_accuracy": val_accuracy,
                        "weighted_score": composite},
                       args.output_dir / "student_best_checkpoint.pt")

        def save_generation_outputs(current_model, generation, action):
            outputs = _timed_greedy_prompts(
                current_model, tokenizer, generation_prompts, context=generation_context,
                new_tokens=24, device=device
            )
            params = list(getattr(action, "params", []) or []) if action is not None else []
            action_label = (f"{type(action).__name__}: {params[0]} -> {params[1]}"
                            if action is not None and len(params) > 1 else None)
            generation_record = {
                "generation": generation, "action": action_label,
                "parameter_count": _params(current_model),
                "elapsed_seconds": outputs["elapsed_seconds"],
                "outputs": outputs["outputs"],
            }
            generation_progress.append(generation_record)
            generation_output_payload["generations"] = generation_progress
            generation_output_payload["completed_generations"] = len(generation_progress)
            generation_outputs_path.write_text(json.dumps(generation_output_payload, indent=2),
                                               encoding="utf-8")
            progress_payload = {
                "status": "training_in_progress",
                "completed_generation_search": generation + 1,
                "planned_generations": limits["generations"],
                "planned_epochs_per_generation": limits["epochs"],
                "parameters": generation_record["parameter_count"],
                "history": progress_history,
                "generation_examples": generation_progress,
            }
            (args.output_dir / "progress.json").write_text(
                json.dumps(progress_payload, indent=2), encoding="utf-8"
            )
            print(f"Generation {generation} output examples after {action_label or 'no action'}:",
                  flush=True)
            for item in outputs["outputs"]:
                print(f"  {item['prompt']} -> {item['generated_text']!r}", flush=True)

        start = time.perf_counter()
        trained, history = train_generations(traced.gm, train_loader, val_loader, cfg,
                                            sim_train_loader=train_loader, sim_val_loader=val_loader,
                                            generation_callback=save_best_student,
                                            post_action_callback=save_generation_outputs)
        elapsed = time.perf_counter() - start
        final_params = _params(trained)
        print("Timing the trained student on the same 10 prompts ...", flush=True)
        benchmark_after = _timed_greedy_prompts(
            trained, tokenizer, benchmark_prompts, context=limits["context"],
            new_tokens=limits["new"], device=device
        )
        benchmark_agreement = _benchmark_output_agreement(benchmark_before, benchmark_after)
        print(f"Final benchmark: {benchmark_after['elapsed_seconds']:.2f}s total; "
              f"{benchmark_after['seconds_per_prompt']:.2f}s/prompt; "
              f"same-prompt token agreement={benchmark_agreement['token_accuracy']:.3f}", flush=True)
        generation_metrics = _generation_agreement(
            trained, val_snapshot, context=limits["context"],
            pad_token_id=tokenizer.pad_token_id, device=device
        )
        final_generation_outputs = generation_progress[-1] if generation_progress else {
            "elapsed_seconds": benchmark_after["elapsed_seconds"],
            "outputs": benchmark_after["outputs"],
        }
        extended_before = {**original_size_examples,
                           "outputs": original_generation_outputs}
        extended_after = {"elapsed_seconds": final_generation_outputs["elapsed_seconds"],
                          "prompt_count": len(generation_prompts),
                          "outputs": final_generation_outputs["outputs"]}
        extended_agreement = _benchmark_output_agreement(extended_before, extended_after)
        generation_output_payload["status"] = "completed"
        generation_output_payload["completed_generations"] = len(generation_progress)
        generation_outputs_path.write_text(json.dumps(generation_output_payload, indent=2),
                                           encoding="utf-8")
        report = {
            "experiment": "010_gpt2_compression", "model": args.model,
            "run_mode": "smoke" if args.smoke else "larger", "device": str(device), "seed": seed,
            "planned_generations": limits["generations"], "epochs_per_generation": limits["epochs"],
            "dataset": {"train_prompt_count": len(train_prompts),
                        "validation_prompt_count": len(validation_prompts),
                        "train_target_tokens": len(train_targets),
                        "validation_target_tokens": len(val_targets),
                        "context_tokens": limits["context"],
                        "max_teacher_continuation_tokens": limits["new"],
                        "teacher_reference_top_k": 8,
                        "tokenizer": {"name": args.model,
                                      "pad_token_id": tokenizer.pad_token_id,
                                      "eos_token_id": tokenizer.eos_token_id,
                                      "vocabulary_size": len(tokenizer)},
                        "student_initialized_from_teacher_checkpoint": True,
                        "train_examples": len(train_ds), "validation_examples": len(val_ds)},
            "capture": {"route": "hf_symbolic_editable", "nodes": len(list(traced.gm.graph.nodes)),
                        "parameter_count": captured_params},
            "objective_weights": {"teacher_token_agreement": 0.1, "model_size": 1.0},
            "baseline_teacher_token_accuracy": baseline_acc,
            "eligible_shrink_actions": eligible_count, "evaluated_shrink_actions": len(actions),
            "eligible_candidate_targets": eligible_candidate_targets,
            "candidate_memory_budget": candidate_budget,
            "offered_candidate_targets": [str(action.params[0]) for action in actions],
            "rejected_candidates": rejected_candidates,
            "initial_parameters": initial_params,
            "captured_parameters": captured_params, "final_parameters": final_params,
            "parameter_reduction_fraction": 1.0 - final_params / max(initial_params, 1),
            "heldout_teacher_token_accuracy": history["val_acc"][-1],
            "heldout_teacher_token_loss": history["val_loss"][-1],
            "training_teacher_token_accuracy": history["train_acc"][-1],
            "heldout_greedy_generation_agreement": generation_metrics,
            "initial_capture_teacher_generation_agreement": initial_capture_agreement,
            "generation_examples": {
                "prompts": generation_prompts,
                "original_outputs": original_generation_outputs,
                "generations": generation_progress,
            },
            "same_prompt_generation_benchmark": {
                "prompts": benchmark_prompts, "before_training": benchmark_before,
                "after_training": benchmark_after, "output_agreement": benchmark_agreement,
            },
            "extended_generation_examples": {
                "new_tokens_requested": 24,
                "context_limit": generation_context,
                "before_training": extended_before,
                "after_training": extended_after,
                "output_agreement": extended_agreement,
            },
            "score_calibration": {"parameter_count_scale": 1.0 / max(1, captured_params),
                                  "baseline_size_score": 0.5,
                                  "final_size_score": score_count_weights(trained, cfg),
                                  "weight_denominator": 1.1},
            "best_checkpoint": {"file": "student_best_checkpoint.pt", **best},
            "simulation_score_evaluations": score_evaluations,
            "training_seconds": elapsed, "history": history,
            "selected_actions": dict(Counter(history.get("action_name", []))),
            "action_counts_by_type": {
                "eligible": {"DelNeurons": eligible_count},
                "offered_to_search": {"DelNeurons": len(actions)},
                "selected": dict(Counter(history.get("action_name", []))),
            },
            "enabled_action_types": ["DelNeurons"],
            "status": "shrunk" if final_params < initial_params else "no_shrink_selected",
        }
        chart_path = args.output_dir / "training_history.png"
        if _write_training_history_chart(report, chart_path):
            report["training_history_chart"] = chart_path.name
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output_dir / "report.md").write_text(_render_markdown_report(report), encoding="utf-8")
        (args.output_dir / "progress.json").write_text(json.dumps({
            "status": "completed", "completed_generations": len(history["generation"]),
            "completed_generation_search": len(generation_progress),
            "planned_generations": limits["generations"],
            "planned_epochs_per_generation": limits["epochs"],
            "parameters": final_params, "training_seconds": elapsed,
            "generation_examples": generation_progress,
            "report": "report.json",
        }, indent=2), encoding="utf-8")
        torch.save({"state_dict": trained.state_dict(), "graph_code": trained.code,
                    "parameter_count": final_params}, args.output_dir / "student_compressed.pt")
        torch.save({"seed": seed,
                    "context_tokens": limits["context"],
                    "max_teacher_continuation_tokens": limits["new"],
                    "teacher_reference_top_k": 8,
                    "tokenizer": {"name": args.model,
                                  "pad_token_id": tokenizer.pad_token_id,
                                  "eos_token_id": tokenizer.eos_token_id,
                                  "vocabulary_size": len(tokenizer)},
                    "context_token_ids": {"train": train_ids, "validation": val_ids},
                    "attention_masks": {"train": train_masks, "validation": val_masks},
                    "teacher_targets": {"train": train_targets, "validation": val_targets},
                    "teacher_reference_outputs": {"train": train_snapshot,
                                                   "validation": val_snapshot},
                    "prompt_partitions": {"train": train_prompts,
                                          "validation": validation_prompts}},
                   args.output_dir / "teacher_memory_snapshot.pt")
        print(f"Status: {report['status']}; parameters {initial_params:,} -> {final_params:,}; "
              f"eligible shrink actions={eligible_count}; evaluated={len(actions)}; "
              f"held-out teacher-token accuracy={history['val_acc'][-1]:.3f}; "
              f"held-out generation agreement={generation_metrics['token_accuracy']:.3f}; "
              f"same-prompt benchmark {benchmark_before['elapsed_seconds']:.2f}s -> "
              f"{benchmark_after['elapsed_seconds']:.2f}s; "
              f"report={args.output_dir / 'report.json'}", flush=True)
    finally:
        project_config.WEIGHT_COUNT_WEIGHT = old_size_weight
        lr_config.MIN_LEARNING_RATE = previous_min_lr


if __name__ == "__main__":
    main()
