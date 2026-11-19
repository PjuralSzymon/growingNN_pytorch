"""Experiment 011: replay Exp010 shrink actions with zero retraining.

This pass replays the exact nine DelNeurons targets selected in the completed
Exp010 run, starting from Exp010's untouched captured checkpoint. It performs
no optimizer steps, gradient descent, or MCTS scoring. It generates and saves
the same three 24-token examples after each action, then compares those outputs
to both the original teacher and Exp010's trained-after-action outputs.

Run from the repository root::

    python ./experiments/run_llm_exp011_gpt2_action_only.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_llm_exp010_gpt2_compression import (
    FixedOutputProjection,
    _benchmark_output_agreement,
    _params,
    _timed_greedy_prompts,
)
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.traced_model import TracedModel
from growingnn.training.causal_lm import CausalLMNextTokenAdapter, pack_token_inputs
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph

DEFAULT_RUN = ROOT / "experiments/output/train_llm/exp010_gpt2_compression/g10_e10"
DEFAULT_OUTPUT = ROOT / "experiments/output/train_llm/exp011_gpt2_action_only"


def _write_chart(rows: list[dict], path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    generations = [row["generation"] for row in rows]
    action_only = [row["action_only_vs_teacher"]["token_accuracy"] for row in rows]
    combined = [row["exp010_trained_vs_teacher"]["token_accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(generations, action_only, marker="o", label="Action only, no retraining")
    ax.plot(generations, combined, marker="s", label="Exp010: action + retraining")
    ax.set_xlabel("Action generation")
    ax.set_ylabel("Token agreement with original GPT-2")
    ax.set_ylim(0, 1.05)
    ax.set_title("Separating shrink actions from retraining")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _write_parameter_chart(initial_parameters: int, rows: list[dict], path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = ["Initial"] + [str(row["generation"]) for row in rows]
    parameters = [initial_parameters] + [int(row["parameter_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(range(len(labels)), [value / 1_000_000 for value in parameters], marker="o",
            color="#2563eb", linewidth=2)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Action generation (Initial = before any action)")
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("GPT-2 parameter count during action-only replay")
    ax.grid(alpha=0.25)
    for index, value in enumerate(parameters):
        ax.annotate(f"{value / 1_000_000:.2f}M", (index, value / 1_000_000),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _render_markdown(report: dict) -> str:
    first = report["generations"][0]
    first_action_agreement = first["action_only_vs_teacher"]["token_accuracy"]
    first_combined_agreement = first["exp010_trained_vs_teacher"]["token_accuracy"]
    first_matches = first["action_only_vs_teacher"]["matched_tokens"]
    first_total = first["action_only_vs_teacher"]["total_tokens"]
    lines = [
        "# Experiment 011: GPT-2 shrink actions without retraining", "",
        "## Purpose", "",
        "Experiment 010 recorded outputs after a generation's training and shrink action. This replay isolates the action sequence: it starts from the same untouched checkpoint, applies the same actions in the same order, and does not train between or after actions.", "",
        "## Run facts", "",
        f"- **Optimizer steps:** {report['optimizer_steps']}",
        f"- **Training epochs:** {report['training_epochs']}",
        f"- **Replayed actions:** {len(report['generations'])}",
        f"- **Initial FX capture vs teacher:** {report['initial_capture_vs_teacher']['token_accuracy']:.1%} token agreement",
        f"- **Parameters:** {report['initial_parameters']:,} → {report['final_parameters']:,} ({report['parameter_reduction_fraction']:.2%} reduction)",
        f"- **Action-only replay time:** {report['elapsed_seconds']:.1f}s", "",
        "The action-only line shows the effect of the structural edits on the original weights. The Experiment 010 line includes its retraining and is copied from the saved per-generation outputs; Experiment 010 was not rerun.", "",
        "## Finding", "",
        f"The first shrink action by itself, applied to the untouched weights with zero training, reduced agreement with original GPT-2 to **{first_matches}/{first_total} tokens ({first_action_agreement:.1%})**. Exp010's trained-plus-action output for that step agreed on **{first_combined_agreement:.1%}**. The deletion alone is therefore sufficient to cause a large generation change on these prompts.", "",
        "This does not prove retraining had no additional effect. Exp010 trained the model before applying that first action, while this replay applies it to untouched weights. A no-action training-only control is still needed to measure retraining's separate contribution.", "",
    ]
    if report.get("chart"):
        lines.extend([f"![Action-only compared with trained outputs by generation]({report['chart']})", ""])
    if report.get("parameter_chart"):
        lines.extend(["## Parameter count by generation", "",
                      f"![GPT-2 parameter count during action-only replay]({report['parameter_chart']})", ""])
    lines.extend([
        "## Output agreement by generation", "",
        "| Generation | Action | Parameters | Action-only vs teacher | Exp010 trained vs teacher | Action-only vs Exp010 |", "| ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in report["generations"]:
        lines.append(
            f"| {row['generation']} | `{row['action_target']}` | {row['parameter_count']:,} | "
            f"{row['action_only_vs_teacher']['token_accuracy']:.1%} | "
            f"{row['exp010_trained_vs_teacher']['token_accuracy']:.1%} | "
            f"{row['action_only_vs_exp010']['token_accuracy']:.1%} |"
        )
    lines.extend(["", "## Examples after each action", ""])
    original = {item["prompt"]: item["generated_text"] for item in report["original_outputs"]}
    for prompt in report["prompts"]:
        lines.extend([f"### {prompt}", "", f"**Original GPT-2:** {original.get(prompt, '')}", "",
                      "| Generation | Action-only output | Exp010 output after training + action |",
                      "| ---: | --- | --- |"])
        for row in report["generations"]:
            action_text = row["action_only_outputs_by_prompt"].get(prompt, "").replace("|", "\\|").replace("\n", " ")
            trained_text = row["exp010_outputs_by_prompt"].get(prompt, "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['generation']} | {action_text} | {trained_text} |")
        lines.append("")
    graph_snapshots = report.get("graph_snapshots", {})
    if graph_snapshots:
        lines.extend(["## FX graph structure snapshots", "",
                      "The diagrams focus on the three Transformer blocks that Exp010 actually changed. The project's filtered FX drawer collapses functional operations and untouched blocks while preserving dependencies between the shown FX module calls. Compact labels include each module's weight shape, so changes in projection width are visible.", ""])
        for key, title in (("initial", "Initial graph"), ("middle", "Middle of action sequence"),
                           ("final", "Final graph after all actions")):
            snapshot = graph_snapshots.get(key)
            if snapshot:
                lines.extend([f"### {title}", "",
                              f"Action generation: {snapshot['generation_label']}", "",
                              f"Parameters: {snapshot['parameter_count']:,}", "",
                              f"![{title}](graphs/{snapshot['file']})", ""])
    lines.extend([
        "## How to interpret this comparison", "",
        "- If the no-training outputs degrade in the same way and at the same actions, the structural deletions are sufficient to explain much of the failure.",
        "- If the no-training outputs stay closer to the teacher while Exp010 diverges, retraining or the interaction between retraining and deletion contributed to the failure.",
        "- If both diverge but by different amounts, both effects matter. This replay alone does not estimate a pure training-only effect; that would require a separate no-action training control.",
        "", "## Limitations", "",
        "This pass replays the exact Exp010 action targets but does not rerun MCTS. Candidate choices in Exp010 were themselves made using trained simulations. The comparison uses the same three 24-token prompts saved by Exp010, so it is diagnostic rather than a broad language-quality evaluation.", "",
    ])
    return "\n".join(lines)


def run(run_dir: Path, output_dir: Path, *, new_tokens: int, device: torch.device,
        local_files_only: bool) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8-sig"))
    outputs_data = json.loads((run_dir / "generation_outputs.json").read_text(encoding="utf-8-sig"))
    if report.get("status") != "shrunk" or outputs_data.get("status") != "completed":
        raise RuntimeError("Experiment 010 must have a completed g10_e10 report and generation outputs")
    targets = [str(target) for target in report.get("history", {}).get("action_from", [])]
    if not targets:
        raise RuntimeError("Experiment 010 report contains no selected action sequence to replay")
    prompts = outputs_data["prompts"]
    original_outputs = outputs_data["original_outputs"]
    trained_generations = outputs_data["generations"]
    if len(trained_generations) < len(targets):
        raise RuntimeError("Experiment 010 generation outputs do not cover all selected actions")

    tokenizer = AutoTokenizer.from_pretrained(report["model"], local_files_only=local_files_only)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        report["model"], torch_dtype=torch.float32, local_files_only=local_files_only,
    ).to(device).eval()
    snapshot = torch.load(run_dir / "teacher_memory_snapshot.pt", map_location=device,
                          weights_only=True)
    sample = pack_token_inputs(
        snapshot["context_token_ids"]["train"][:1],
        snapshot["attention_masks"]["train"][:1],
    ).to(device)
    context_limit = int(getattr(teacher.config, "max_position_embeddings",
                                getattr(teacher.config, "n_positions", 1024)))
    traced = TracedModel.create(
        CausalLMNextTokenAdapter(teacher), tuple(sample.shape),
        capture="hf_symbolic_editable", example_args=(sample,),
    )
    for path, module in list(traced.gm.named_modules()):
        if path and isinstance(module, nn.Linear) and module.out_features == len(tokenizer):
            traced.gm.set_submodule(path, FixedOutputProjection(module))
    traced.shapes()

    offered_targets = {str(target) for target in report.get("offered_candidate_targets", [])}
    eligible = DelNeurons.generate_all_actions(traced, ratio=0.9)
    eligible_targets = {str(action.params[0]) for action in eligible}
    missing = offered_targets - eligible_targets
    if missing:
        raise RuntimeError(f"Saved offered actions cannot be reproduced: {sorted(missing)}")
    for target in eligible_targets - offered_targets:
        module = traced.gm.get_submodule(target)
        traced.gm.set_submodule(target, FixedOutputProjection(module))
    traced.gm.load_state_dict(
        torch.load(run_dir / "student_initial_checkpoint.pt", map_location=device,
                   weights_only=True)["state_dict"], strict=True,
    )
    traced.invalidate()
    traced.shapes()

    output_dir.mkdir(parents=True, exist_ok=True)
    initial_params = _params(traced.gm)
    start = time.perf_counter()
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_snapshots: dict[str, dict] = {}
    changed_blocks = {target.rsplit(".mlp.", 1)[0] for target in targets}
    graph_module_targets = {
        str(node.target)
        for node in traced.gm.graph.nodes
        if node.op == "call_module"
        and any(str(node.target).startswith(block + ".") for block in changed_blocks)
    }

    def save_graph_snapshot(key: str, title: str, generation_label: str) -> None:
        filename = f"gpt2_action_only_{key}.png"
        destination = graphs_dir / filename
        draw_filtered_fx_graph(
            traced.gm, str(destination.with_suffix("")), fmt="png",
            include_functions=False, module_targets=graph_module_targets,
            compact_labels=True, rankdir="LR",
        )
        graph_snapshots[key] = {
            "title": title, "file": filename, "generation_label": generation_label,
            "parameter_count": _params(traced.gm),
            "graph_node_count": len(list(traced.gm.graph.nodes)),
            "displayed_module_count": len(graph_module_targets),
            "changed_block_targets": sorted(changed_blocks),
        }

    print("Saving initial graph snapshot ...", flush=True)
    save_graph_snapshot("initial", "Initial graph", "before any action")
    print("Pass 1: generating the untouched captured model output ...", flush=True)
    initial = _timed_greedy_prompts(
        traced.gm, tokenizer, prompts, context=context_limit, new_tokens=new_tokens, device=device,
    )
    initial_capture = _benchmark_output_agreement({"outputs": original_outputs}, initial)
    (output_dir / "initial_capture.json").write_text(json.dumps({
        "fx_vs_hf_agreement": initial_capture, "outputs": initial["outputs"],
    }, indent=2), encoding="utf-8")
    print(f"  Initial FX capture: {initial_capture['matched_tokens']}/"
          f"{initial_capture['total_tokens']} tokens match Hugging Face", flush=True)

    generation_rows: list[dict] = []
    progress_path = output_dir / "progress.json"
    output_path = output_dir / "generation_outputs.json"
    for index, target in enumerate(targets):
        candidates = DelNeurons.generate_all_actions(traced, ratio=0.9)
        action = next((candidate for candidate in candidates if str(candidate.params[0]) == target), None)
        if action is None:
            raise RuntimeError(f"Action target {target} is no longer eligible at replay generation {index}")
        print(f"Pass 1: action {index + 1}/{len(targets)} {target}; no training ...", flush=True)
        action.execute(traced)
        traced.invalidate()
        traced.shapes()
        generated = _timed_greedy_prompts(
            traced.gm, tokenizer, prompts, context=context_limit,
            new_tokens=new_tokens, device=device,
        )
        original_agreement = _benchmark_output_agreement({"outputs": original_outputs}, generated)
        trained_row = trained_generations[index]
        trained_outputs = {"outputs": trained_row["outputs"]}
        action_vs_trained = _benchmark_output_agreement(trained_outputs, generated)
        row = {
            "generation": index,
            "action_target": target,
            "action_ratio": float(action.params[1]),
            "parameter_count": _params(traced.gm),
            "action_only_vs_teacher": original_agreement,
            "exp010_trained_vs_teacher": _benchmark_output_agreement(
                {"outputs": original_outputs}, trained_outputs,
            ),
            "action_only_vs_exp010": action_vs_trained,
            "elapsed_seconds": generated["elapsed_seconds"],
            "action_only_outputs": generated["outputs"],
            "action_only_outputs_by_prompt": {
                item["prompt"]: item["generated_text"] for item in generated["outputs"]
            },
            "exp010_outputs_by_prompt": {
                item["prompt"]: item["generated_text"] for item in trained_row["outputs"]
            },
        }
        generation_rows.append(row)
        if index == (len(targets) - 1) // 2:
            print("Saving middle graph snapshot ...", flush=True)
            save_graph_snapshot("middle", "Middle graph", f"after action generation {index}")
        partial = {
            "status": "running", "training_epochs": 0, "optimizer_steps": 0,
            "completed_actions": len(generation_rows), "planned_actions": len(targets),
            "generations": generation_rows,
        }
        progress_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        output_path.write_text(json.dumps({
            "status": "running", "prompts": prompts, "original_outputs": original_outputs,
            "generations": generation_rows,
        }, indent=2), encoding="utf-8")
        for item in generated["outputs"]:
            print(f"  {item['prompt']} -> {item['generated_text']!r}", flush=True)
        print(f"  Agreement with original: {original_agreement['token_accuracy']:.1%}; "
              f"parameters={row['parameter_count']:,}", flush=True)

    print("Saving final graph snapshot ...", flush=True)
    save_graph_snapshot("final", "Final graph", f"after action generation {len(targets) - 1}")
    elapsed = time.perf_counter() - start
    final_params = _params(traced.gm)
    full_report = {
        "experiment": "011_gpt2_action_only_replay",
        "status": "completed",
        "source_experiment": str(run_dir),
        "model": report["model"],
        "device": str(device),
        "action_sequence_replayed_from_exp010": targets,
        "capture_route": "hf_symbolic_editable",
        "training_epochs": 0,
        "optimizer_steps": 0,
        "mcts_runs": 0,
        "initial_parameters": initial_params,
        "final_parameters": final_params,
        "parameter_reduction_fraction": 1.0 - final_params / max(1, initial_params),
        "initial_capture_vs_teacher": initial_capture,
        "finding": (
            f"The first deletion alone, with zero training, retained "
            f"{generation_rows[0]['action_only_vs_teacher']['matched_tokens']}/"
            f"{generation_rows[0]['action_only_vs_teacher']['total_tokens']} teacher token "
            f"positions ({generation_rows[0]['action_only_vs_teacher']['token_accuracy']:.1%}) "
            f"on the three prompts. This shows the structural action is sufficient for a large "
            "generation change; it does not quantify retraining's separate contribution."
        ),
        "prompts": prompts,
        "original_outputs": original_outputs,
        "generations": generation_rows,
        "graph_snapshots": graph_snapshots,
        "elapsed_seconds": elapsed,
        "notes": [
            "No optimizer, gradient descent, or MCTS score evaluation was called in this experiment.",
            "The action targets are replayed from Exp010 so this run isolates structural edits from retraining while preserving the exact selected sequence.",
            "The Exp010 comparison line includes its training and is loaded from the saved report artifacts; Exp010 was not rerun.",
        ],
    }
    chart = _write_chart(generation_rows, output_dir / "action_only_vs_exp010.png")
    full_report["chart"] = chart
    full_report["parameter_chart"] = _write_parameter_chart(
        initial_params, generation_rows, output_dir / "parameter_count_by_generation.png",
    )
    markdown_path = output_dir / "report.md"
    markdown_path.write_text(_render_markdown(full_report), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    output_path.write_text(json.dumps({
        "status": "completed", "prompts": prompts, "original_outputs": original_outputs,
        "generations": generation_rows,
    }, indent=2), encoding="utf-8")
    progress_path.write_text(json.dumps({
        "status": "completed", "training_epochs": 0, "optimizer_steps": 0,
        "completed_actions": len(generation_rows), "planned_actions": len(targets),
        "report": "report.json",
    }, indent=2), encoding="utf-8")
    print(f"\nExperiment 011 complete; no training was performed.", flush=True)
    print(f"Parameters: {initial_params:,} -> {final_params:,}; "
          f"reduction={full_report['parameter_reduction_fraction']:.2%}", flush=True)
    print(f"Report: {markdown_path}", flush=True)
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run(args.run_dir, args.output_dir, new_tokens=args.new_tokens, device=device,
        local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
