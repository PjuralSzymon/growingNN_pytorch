"""Rebuild the readable GPT-2 Experiment 010 report from its saved run files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_llm_exp010_gpt2_compression import (
    FixedOutputProjection,
    _hf_greedy_prompts,
    _benchmark_output_agreement,
    _render_markdown_report,
    _timed_greedy_prompts,
)
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.traced_model import TracedModel
from growingnn.training.causal_lm import CausalLMNextTokenAdapter, pack_token_inputs


RUN_DIR = ROOT / "experiments" / "output" / "train_llm" / "exp010_gpt2_compression" / "g10_e10"


def regenerate(run_dir: Path = RUN_DIR, *, new_tokens: int = 24) -> Path:
    """Reconstruct initial/final students, add longer examples, and render reports."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("status") != "shrunk":
        raise RuntimeError(f"run is not a completed shrink run: {report.get('status')}")
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(report["model"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        report["model"], torch_dtype=torch.float32
    ).to(device).eval()
    teacher_payload = torch.load(run_dir / "teacher_checkpoint.pt", map_location="cpu",
                                 weights_only=True)
    teacher.load_state_dict(teacher_payload["state_dict"], strict=True)
    generation_context = int(getattr(teacher.config, "max_position_embeddings",
                                     getattr(teacher.config, "n_positions", 1024)))
    snapshot = torch.load(run_dir / "teacher_memory_snapshot.pt", map_location="cpu",
                          weights_only=True)
    sample = pack_token_inputs(
        snapshot["context_token_ids"]["train"][:1],
        snapshot["attention_masks"]["train"][:1],
    ).to(device)
    prompts = report["same_prompt_generation_benchmark"]["prompts"][:3]
    original_outputs = _hf_greedy_prompts(
        teacher, tokenizer, prompts, context=generation_context,
        new_tokens=new_tokens, device=device
    )
    wrapped = CausalLMNextTokenAdapter(teacher)
    traced = TracedModel.create(
        wrapped, tuple(sample.shape), capture="hf_symbolic_editable", example_args=(sample,)
    )
    for path, module in list(traced.gm.named_modules()):
        if path and isinstance(module, nn.Linear) and module.out_features == len(tokenizer):
            traced.gm.set_submodule(path, FixedOutputProjection(module))
    traced.shapes()

    offered = set(report["offered_candidate_targets"])
    eligible = DelNeurons.generate_all_actions(traced, ratio=0.9)
    eligible_targets = {action.params[0] for action in eligible}
    missing = offered - eligible_targets
    if missing:
        raise RuntimeError(f"saved offered candidates are not reproducible: {sorted(missing)}")
    for target in eligible_targets - offered:
        module = traced.gm.get_submodule(target)
        traced.gm.set_submodule(target, FixedOutputProjection(module))
    actions = DelNeurons.generate_all_actions(traced, ratio=0.9)
    traced.gm.load_state_dict(
        torch.load(run_dir / "student_initial_checkpoint.pt", map_location="cpu",
                   weights_only=True)["state_dict"], strict=True
    )

    print(f"Generating {new_tokens}-token examples from the saved initial checkpoint ...", flush=True)
    before = _timed_greedy_prompts(
        traced.gm, tokenizer, prompts, context=generation_context,
        new_tokens=new_tokens, device=device,
    )
    report["initial_capture_teacher_generation_agreement"] = _benchmark_output_agreement(
        {"outputs": original_outputs}, before
    )
    for target in report["history"].get("action_from", []):
        candidates = DelNeurons.generate_all_actions(traced, ratio=0.9)
        action = next((candidate for candidate in candidates if candidate.params[0] == target), None)
        if action is None:
            raise RuntimeError(f"could not replay saved DelNeurons action for {target}")
        action.execute(traced)

    compressed = torch.load(run_dir / "student_compressed.pt", map_location="cpu",
                            weights_only=True)
    traced.gm.load_state_dict(compressed["state_dict"], strict=True)
    print(f"Generating {new_tokens}-token examples from the saved compressed checkpoint ...", flush=True)
    after = _timed_greedy_prompts(
        traced.gm, tokenizer, prompts, context=generation_context,
        new_tokens=new_tokens, device=device,
    )
    report["extended_generation_examples"] = {
        "new_tokens_requested": new_tokens, "context_limit": generation_context,
        "before_training": {**before, "outputs": original_outputs},
        "after_training": after,
        "output_agreement": _benchmark_output_agreement(before, after),
    }
    chart_path = run_dir / "training_history.png"
    report["training_history_chart"] = chart_path.name if chart_path.exists() else None
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = run_dir / "report.md"
    markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")
    print(f"Report regenerated: {markdown_path}", flush=True)
    return markdown_path


if __name__ == "__main__":
    regenerate()
