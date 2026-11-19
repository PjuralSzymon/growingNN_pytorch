"""GPT-2 end-to-end load, HF-FX trace, fidelity, timing, and graph-drawing regression.

Run from the repository root::

    python ./tests/regression/actions/llm_capture_actions_regression.py

Loads unquantized float32 GPT-2, runs the same prompt through the original and
the FX GraphModule held by TracedModel, compares logits and greedy output, and
writes the full and simplified FX graph PNGs and smoke-tests actions on copies.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from growingnn.actions.add_neurons import AddNeurons
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.action import Layer_Type
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.traced_model import TracedModel
from growingnn.actions.utils.layer_resize import linear_output_resize_rejection_reason
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph


MODEL_ID = "openai-community/gpt2"
DEFAULT_OUTPUT_DIR = Path("testResults/regression/gpt2_fidelity")


class GPT2Logits(nn.Module):
    """Expose a stable tensor-only interface for both the original and FX graph."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(forward: Callable[[torch.Tensor], torch.Tensor], input_ids: torch.Tensor,
               *, warmup: int, repeats: int, device: torch.device) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward(input_ids)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            forward(input_ids)
        _sync(device)
    elapsed = time.perf_counter() - start
    return {"total_seconds": elapsed, "mean_seconds": elapsed / repeats,
            "runs_per_second": repeats / elapsed if elapsed else float("inf")}


def _greedy_generate(forward: Callable[[torch.Tensor], torch.Tensor], input_ids: torch.Tensor,
                     *, max_new_tokens: int, eos_token_id: int | None) -> torch.Tensor:
    sequence = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = forward(sequence)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
    return sequence


def _parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    count = 0
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            count += parameter.numel()
    return count


def _graph_summary(gm: fx.GraphModule) -> dict:
    nodes = list(gm.graph.nodes)
    module_targets = Counter(str(node.target) for node in nodes if node.op == "call_module")
    return {
        "node_count": len(nodes),
        "node_kinds": dict(Counter(node.op for node in nodes)),
        "module_call_count": sum(node.op == "call_module" for node in nodes),
        "unique_module_targets": len(module_targets),
        "module_targets": dict(module_targets),
        "attention_targets": [target for target in module_targets if any(
            term in target.lower() for term in ("attn", "attention", "c_attn", "q_attn")
        )],
        "mlp_targets": [target for target in module_targets if any(
            term in target.lower() for term in ("mlp", "c_fc", "fc")
        )],
    }


def _draw_graphs(gm: fx.GraphModule, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_png": output_dir / "gpt2_fx_full.png",
        "simplified_png": output_dir / "gpt2_fx_simplified.png",
        "full_pdf": output_dir / "gpt2_fx_full.pdf",
        "simplified_pdf": output_dir / "gpt2_fx_simplified.pdf",
    }
    draw_torch_fx_graph(gm, str(paths["full_png"].with_suffix("")), fmt="png")
    draw_filtered_fx_graph(gm, str(paths["simplified_png"].with_suffix("")), fmt="png")
    draw_torch_fx_graph(gm, str(paths["full_pdf"].with_suffix("")), fmt="pdf")
    draw_filtered_fx_graph(gm, str(paths["simplified_pdf"].with_suffix("")), fmt="pdf")
    return {key: str(path.resolve()) for key, path in paths.items()}


def _action_generators():
    return [
        ("AddResLinearLayer (EYE + ZERO)", lambda traced: AddResLinearLayer.generate_all_actions(
            traced, layer_types=(Layer_Type.EYE, Layer_Type.ZERO))),
        ("AddResConvLayer", AddResConvLayer.generate_all_actions),
        ("AddSeqLinearLayer", AddSeqLinearLayer.generate_all_actions),
        ("AddSeqConvLayer (plain)", lambda traced: AddSeqConvLayer.generate_all_actions(
            traced, add_batchnorm_and_relu=False, before_global_pool=True)),
        ("AddSeqConvLayer (BatchNorm + ReLU)", lambda traced: AddSeqConvLayer.generate_all_actions(
            traced, add_batchnorm_and_relu=True, before_global_pool=True)),
        ("AddSeqDropoutLayer (p=0.1)", lambda traced: AddSeqDropoutLayer.generate_all_actions(traced, p=0.1)),
        ("AddSeqDropoutLayer (p=0.2)", lambda traced: AddSeqDropoutLayer.generate_all_actions(traced, p=0.2)),
        ("AddSeqDropoutLayer (p=0.5)", lambda traced: AddSeqDropoutLayer.generate_all_actions(traced, p=0.5)),
        ("AddNeurons (ratio=1.1)", lambda traced: AddNeurons.generate_all_actions(traced, ratio=1.1)),
        ("AddNeurons (ratio=1.5)", lambda traced: AddNeurons.generate_all_actions(traced, ratio=1.5)),
        ("AddNeurons (ratio=2.0)", lambda traced: AddNeurons.generate_all_actions(traced, ratio=2.0)),
        ("DelLayer", DelLayer.generate_all_actions),
        ("DelNeurons (ratio=0.1)", lambda traced: DelNeurons.generate_all_actions(traced, ratio=0.1)),
        ("DelNeurons (ratio=0.5)", lambda traced: DelNeurons.generate_all_actions(traced, ratio=0.5)),
        ("DelNeurons (ratio=0.9)", lambda traced: DelNeurons.generate_all_actions(traced, ratio=0.9)),
    ]


def _generate_action_summary(traced: TracedModel, output_dir: Path) -> dict:
    variant_counts: dict[str, int | None] = {}
    class_counts: Counter[str] = Counter()
    generated_by_class: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    skipped: dict[str, str] = {}
    validation_actions: dict[str, object] = {}
    for variant, generate in _action_generators():
        print(f"  generating {variant} ...", flush=True)
        try:
            actions = generate(traced)
            variant_counts[variant] = len(actions)
            print(f"    generated {len(actions)}", flush=True)
            for action in actions:
                class_name = type(action).__name__
                class_counts[class_name] += 1
                generated_by_class.setdefault(class_name, []).append(str(action))
                existing = validation_actions.get(class_name)
                is_fx_candidate = len(action.params) > 4 and action.params[4] in (
                    "fx_edge", "fx_residual"
                )
                existing_is_fx = (
                    existing is not None
                    and len(existing.params) > 4
                    and existing.params[4] in ("fx_edge", "fx_residual")
                )
                if existing is None or (is_fx_candidate and not existing_is_fx):
                    validation_actions[class_name] = action
        except Exception as exc:
            variant_counts[variant] = 0
            errors[variant] = f"{type(exc).__name__}: {exc}"

    review_deletions = [
        description
        for description in generated_by_class.get("DelLayer", [])
        if any(term in description.lower() for term in ("wpe", "wte", "c_proj", "embedding"))
    ]

    total = sum(count for count in variant_counts.values() if count is not None)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "gpt2_action_counts.png"
    labels = [label for label, count in variant_counts.items() if count is not None]
    values = [variant_counts[label] for label in labels]
    fig_height = max(5, len(labels) * 0.32)
    fig, axis = plt.subplots(figsize=(12, fig_height))
    positions = list(range(len(labels)))
    axis.barh(positions, values, color="#4472C4")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Generated action proposals")
    axis.set_title(f"GPT-2 action proposals by generator variant (total: {total})")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    return {
        "total_action_proposals": total,
        "counts_by_variant": variant_counts,
        "counts_by_action_class": dict(class_counts),
        "generated_action_descriptions": generated_by_class,
        "generator_errors": errors,
        "skipped_action_generators": skipped,
        "deletion_candidates_requiring_semantic_review": review_deletions,
        "_validation_actions": validation_actions,
        "chart_png": str(chart_path.resolve()),
    }


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    print(f"Loading {MODEL_ID} as unquantized float32 on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).eval().to(device)
    original = GPT2Logits(base_model).eval()

    tokens = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    if tokens.shape[1] == 0:
        raise ValueError("Tokenizer returned an empty prompt")
    print(f"Prompt: {args.prompt!r}", flush=True)
    print(f"Input token IDs: {tokens[0].tolist()}", flush=True)

    print("Capturing with Hugging Face HFTracer and native-rank GPT-2 projections ...", flush=True)
    traced = TracedModel.create(
        original,
        tuple(tokens.shape),
        capture="hf_symbolic_editable",
        example_args=(tokens,),
    )
    traced.gm.eval()
    graph_forward = traced.gm
    traced.shapes()
    neuron_resize_rejections = {}
    for layer_id in traced.hidden_modules():
        module = traced.gm.get_submodule(layer_id)
        if not isinstance(module, nn.Linear):
            continue
        requested_width = max(1, int(module.out_features * 1.1))
        reason = linear_output_resize_rejection_reason(
            traced.gm, layer_id, requested_width
        )
        if reason is not None:
            neuron_resize_rejections[layer_id] = reason
    rank3_linear_calls = [
        node.name
        for node in graph_forward.graph.nodes
        if node.op == "call_module"
        and isinstance(graph_forward.get_submodule(str(node.target)), nn.Linear)
        and node.args
        and isinstance(node.args[0], fx.Node)
        and getattr(node.args[0].meta.get("tensor_meta"), "shape", None) is not None
        and len(node.args[0].meta["tensor_meta"].shape) == 3
    ]
    if not rank3_linear_calls:
        raise AssertionError("Editable GPT-2 capture did not preserve rank-3 Linear interfaces")
    print(f"Native-rank Linear projection calls: {len(rank3_linear_calls)}", flush=True)
    print(f"Neuron-width candidates rejected with reasons: {len(neuron_resize_rejections)}", flush=True)

    with torch.inference_mode():
        logits_original = original(tokens)
        logits_traced = graph_forward(tokens)
    if not isinstance(logits_traced, torch.Tensor):
        raise TypeError(f"Traced graph returned {type(logits_traced).__name__}, expected Tensor logits")
    difference = (logits_original - logits_traced).abs()
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    allclose = bool(torch.allclose(logits_original, logits_traced, rtol=1e-5, atol=1e-6))

    eos_id = tokenizer.eos_token_id
    generated_original = _greedy_generate(
        original, tokens, max_new_tokens=args.max_new_tokens, eos_token_id=eos_id
    )
    generated_traced = _greedy_generate(
        graph_forward, tokens, max_new_tokens=args.max_new_tokens, eos_token_id=eos_id
    )
    text_original = tokenizer.decode(generated_original[0], skip_special_tokens=True)
    text_traced = tokenizer.decode(generated_traced[0], skip_special_tokens=True)
    generated_ids_match = bool(torch.equal(generated_original, generated_traced))

    print(f"Original output: {text_original!r}", flush=True)
    print(f"Traced output:   {text_traced!r}", flush=True)
    print(f"Logits allclose: {allclose}; max_abs={max_abs_error:.8g}; mean_abs={mean_abs_error:.8g}", flush=True)
    print(f"Greedy token IDs match: {generated_ids_match}", flush=True)

    print("Generating existing mutation actions ...", flush=True)
    graph_nodes_before_actions = len(list(graph_forward.graph.nodes))
    action_summary = _generate_action_summary(traced, args.output_dir)
    validation_actions = action_summary.pop("_validation_actions")
    graph_nodes_after_actions = len(list(graph_forward.graph.nodes))
    traced_parameter_count_before_validation = _parameter_count(graph_forward)
    graph_summary_before_validation = _graph_summary(graph_forward)
    for variant, count in action_summary["counts_by_variant"].items():
        if count is None:
            print(f"  {variant}: not run (see skipped_action_generators)", flush=True)
        else:
            print(f"  {variant}: {count}", flush=True)
    print("Counts by action class:", flush=True)
    for action_class, count in action_summary["counts_by_action_class"].items():
        print(f"  {action_class}: {count}", flush=True)
    for variant, error in action_summary["generator_errors"].items():
        print(f"  {variant}: ERROR {error}", flush=True)
    for variant, reason in action_summary["skipped_action_generators"].items():
        print(f"  skipped {variant}: {reason}", flush=True)
    print(f"Total proposals: {action_summary['total_action_proposals']}", flush=True)
    print(f"Action chart: {action_summary['chart_png']}", flush=True)

    graph_paths = _draw_graphs(graph_forward, args.output_dir)
    print("Graph images:", flush=True)
    for name, path in graph_paths.items():
        print(f"  {name}: {path}", flush=True)

    print("Benchmarking full-prompt forward pass ...", flush=True)
    original_speed = _benchmark(
        original, tokens, warmup=args.warmup, repeats=args.repeats, device=device
    )
    traced_speed = _benchmark(
        graph_forward, tokens, warmup=args.warmup, repeats=args.repeats, device=device
    )

    # Smoke-test representative graph actions on independent CPU copies so each
    # candidate is evaluated against the same captured graph.
    action_validation: dict[str, dict[str, object]] = {}
    validation_tokens = tokens.cpu()
    for class_name in (
        "AddResLinearLayer", "AddSeqLinearLayer", "AddSeqDropoutLayer", "AddNeurons"
    ):
        action = validation_actions.get(class_name)
        if action is None:
            action_validation[class_name] = {"status": "no_candidate"}
            continue
        try:
            validation_gm = copy.deepcopy(traced.gm).cpu()
            validation_traced = TracedModel.create(
                validation_gm,
                tuple(validation_tokens.shape),
                example_args=(validation_tokens,),
            )
            preserve_output = class_name in {
                "AddResLinearLayer", "AddSeqLinearLayer", "AddSeqDropoutLayer"
            }
            with torch.inference_mode():
                baseline_output = validation_gm(validation_tokens)
            copy.deepcopy(action).execute(validation_traced)
            with torch.inference_mode():
                output = validation_traced.gm(validation_tokens)
            max_abs_error = float((baseline_output - output).abs().max().item())
            output_matches = bool(torch.allclose(
                baseline_output, output, rtol=1e-5, atol=1e-6
            ))
            if preserve_output and not output_matches:
                raise AssertionError(
                    f"{class_name} changed the logits despite its identity/zero initialization "
                    f"(max_abs_error={max_abs_error:.8g})"
                )
            action_validation[class_name] = {
                "status": "executed",
                "output_shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all()),
                "preserves_initial_output": output_matches if preserve_output else None,
                "max_abs_output_difference": max_abs_error,
            }
        except Exception as exc:
            action_validation[class_name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    print(f"Representative action validation: {action_validation}", flush=True)

    report = {
        "model_id": MODEL_ID,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": str(device),
        "dtype": str(next(base_model.parameters()).dtype),
        "quantization": None,
        "capture_route": "hf_symbolic_editable",
        "prompt": args.prompt,
        "input_token_ids": tokens[0].tolist(),
        "input_shape": list(tokens.shape),
        "logits_shape": list(logits_original.shape),
        "parameter_count_original": _parameter_count(original),
        "parameter_count_traced": traced_parameter_count_before_validation,
        "parameter_counts_match": _parameter_count(original) == traced_parameter_count_before_validation,
        "graph": graph_summary_before_validation,
        "native_rank3_linear_call_count": len(rank3_linear_calls),
        "neuron_resize_rejections": neuron_resize_rejections,
        "action_generation": action_summary,
        "representative_action_validation": action_validation,
        "graph_node_count_before_action_generation": graph_nodes_before_actions,
        "graph_node_count_after_action_generation": graph_nodes_after_actions,
        "action_generation_mutated_graph": graph_nodes_before_actions != graph_nodes_after_actions,
        "fidelity": {
            "logits_allclose": allclose,
            "logits_max_abs_error": max_abs_error,
            "logits_mean_abs_error": mean_abs_error,
            "original_generated_text": text_original,
            "traced_generated_text": text_traced,
            "generated_token_ids_match": generated_ids_match,
        },
        "timing": {
            "warmup_runs": args.warmup,
            "measured_runs": args.repeats,
            "original": original_speed,
            "traced": traced_speed,
            "traced_over_original": traced_speed["mean_seconds"] / original_speed["mean_seconds"],
        },
        "graphs": graph_paths,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "gpt2_fidelity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Parameters: {report['parameter_count_original']:,} original / {report['parameter_count_traced']:,} traced", flush=True)
    print(f"Mean latency: {original_speed['mean_seconds']:.6f}s original / {traced_speed['mean_seconds']:.6f}s traced", flush=True)
    print(f"Report: {report_path.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    cli = _parse_args()
    try:
        run(cli)
    except Exception as exc:
        error_path = cli.output_dir / "gpt2_fidelity_error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8"
        )
        print(f"Regression failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Details: {error_path.resolve()}", file=sys.stderr)
        raise
