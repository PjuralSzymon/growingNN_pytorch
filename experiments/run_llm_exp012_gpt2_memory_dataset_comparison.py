r"""Run the four-arm GPT-2 memory dataset comparison.

Run from the repository root:
  python .\experiments\run_llm_exp012_gpt2_memory_dataset_comparison.py --smoke
  python .\experiments\run_llm_exp012_gpt2_memory_dataset_comparison.py

The full run fetches WikiText-2 rows from Hugging Face and caches them locally.
Supply --text-file PATH to use a local UTF-8 text corpus instead.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_llm_exp010_gpt2_compression import (
    FixedOutputProjection, _benchmark_output_agreement, _hf_greedy_prompts,
    _timed_greedy_prompts,
)
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.traced_model import TracedModel
from growingnn.training.causal_lm import CausalLMNextTokenAdapter, pack_token_inputs
from growingnn.training.gradient_descent import _evaluate, gradient_descent
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode

MODEL = "openai-community/gpt2"
OUTPUT = ROOT / "experiments/output/train_llm/exp012_gpt2_memory_dataset_comparison"
ACTION_TARGET, ACTION_RATIO = "model.transformer.h.2.mlp.c_fc", 0.9
SEED, CONTEXT, TOP_K = 12012, 32, 8
TEMPLATES = (
    "Explain {topic} in clear steps.", "Write a short factual description of {topic}.",
    "What could happen next in a story about {topic}?",
    "Describe one challenge and one solution involving {topic}.",
    "Give a beginner-friendly example about {topic}.",
    "Summarize why {topic} matters to a community.",
    "Imagine a careful plan for {topic}.", "What are two useful facts about {topic}?",
    "Tell a short story involving {topic}.",
)
TOPICS = (
    "a river crossing a dry valley", "a scientist testing a new material",
    "a small language model", "a city preparing for winter", "a robot learning a tune",
    "a telescope finding a distant planet", "a careful weather forecast", "an old library",
    "a team restoring a wetland", "a sequence of sensor readings", "a camera describing a street",
    "a student building a simple program", "a discovery in the ocean", "a train in the mountains",
    "a garden adapting to less water", "a museum preserving a fragile object",
    "a new method for storing energy", "a community planning a public space",
    "a satellite measuring forest growth", "a musician composing with code",
    "a historian comparing two letters", "a medical device monitoring motion",
    "a chef adapting a family recipe", "a researcher studying animal migration",
    "a factory reducing material waste", "a school using a new learning tool",
    "a spacecraft preparing for a long journey", "a team analyzing economic trends",
    "a village responding to a power outage", "a map of changing coastlines",
    "a bicycle sharing network", "a coastal flood warning", "a forest recovering after fire",
    "a small business managing inventory", "a community science project",
    "a river sensor detecting pollution", "a classroom learning about planets",
    "a home reducing energy use", "a weather balloon collecting data",
    "a city planting shade trees", "a farm using soil sensors", "an ocean buoy measuring waves",
    "a hospital scheduling appointments", "a robot sorting recycled materials",
    "a team mapping local wildlife", "a student designing a board game",
    "a village sharing solar power", "a scientist studying sleep", "a drone checking a bridge",
    "a computer detecting unusual activity", "a town improving bus routes",
    "a lab measuring air quality", "a crew repairing a satellite",
    "a team tracking a migrating bird", "a student learning about sound",
    "a city preparing for a heat wave",
    "a museum restoring a painting", "a city tracking traffic noise",
    "a research team monitoring glaciers", "a family growing herbs indoors",
    "a school planning a science fair", "a mechanic diagnosing an engine",
    "a park ranger protecting nesting birds", "a team designing accessible signs",
    "a weather station on a mountain", "a community organizing a food market",
    "a student building a weather app", "a harbor monitoring water quality",
    "a clinic improving patient check-in", "a farmer testing crop varieties",
    "a library archiving local history", "a team measuring a solar panel",
    "a city planning safer crossings", "a researcher tracking whale calls",
    "a workshop repairing old computers", "a team preparing for an eclipse",
)
EVAL_PROMPTS = [
    "Explain a school using a new learning tool in clear steps.",
    "What could happen next in a story about a garden adapting to less water?",
    "Give a beginner-friendly example related to a spacecraft preparing for a long journey.",
    "Write a short factual description of a satellite measuring forest growth.",
    "Describe one challenge and one solution involving a robot learning a tune.",
    "Explain how a community could respond to a power outage.",
    "Give a simple example about a scientist testing a new material.",
    "What could happen next in a story about a team restoring a wetland?",
    "Write a short description of a medical device monitoring motion.",
    "Explain a new method for storing energy in clear steps.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--text-file", type=Path, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 8 * 1024**3:
        return torch.device("cuda")
    return torch.device("cpu")


def make_prompts(n, seed):
    eval_set = set(EVAL_PROMPTS)
    pool = [t.format(topic=topic) for t in TEMPLATES for topic in TOPICS
            if t.format(topic=topic) not in eval_set]
    if n > len(pool):
        raise ValueError(f"need {n} unique prompts but prompt pool has {len(pool)}")
    return random.Random(seed).sample(pool, n)


def pack_context(prefix, pad_id):
    prefix = prefix[-CONTEXT:]
    missing = CONTEXT - len(prefix)
    return [pad_id] * missing + prefix, [0] * missing + [1] * len(prefix)


def prompt_data(model, tokenizer, prompts, tokens_each, device, batch_size):
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    xs, ms, ys = [], [], []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=CONTEXT).to(device)
        with torch.inference_mode():
            sequences = model.generate(**encoded, do_sample=False, max_new_tokens=tokens_each,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        offset = encoded.input_ids.shape[1]
        for row in range(len(batch)):
            prefix = encoded.input_ids[row][encoded.attention_mask[row].bool()].tolist()
            continuation = sequences[row, offset:].tolist()
            # Force a fixed target budget even when EOS is generated early.
            continuation = continuation[:tokens_each]
            continuation += [int(tokenizer.eos_token_id)] * (tokens_each - len(continuation))
            for token in continuation:
                x, mask = pack_context(prefix, int(tokenizer.pad_token_id))
                xs.append(x)
                ms.append(mask)
                ys.append(int(token))
                prefix.append(int(token))
    return tuple(torch.tensor(v, dtype=torch.long) for v in (xs, ms, ys))


def load_text(text_file, cache_dir):
    if text_file:
        if not text_file.is_file():
            raise FileNotFoundError(text_file)
        rows = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines()
                if len(line.strip()) > 30]
        if len(rows) < 4:
            raise ValueError("local corpus needs at least four non-empty lines")
        split = max(1, int(len(rows) * 0.9))
        return rows[:split], rows[split:], f"local:{text_file.resolve()}"
    # Fetch rows from Hugging Face's Dataset Viewer API. This avoids both the
    # optional datasets/pyarrow packages and Salesforce's legacy AWS archive.
    def fetch_split(split: str, required_chars: int) -> list[str]:
        cache_file = cache_dir / f"wikitext2_{split}_rows.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and sum(map(len, cached)) >= required_chars:
                return cached
        texts: list[str] = []
        offset = 0
        total_rows = None
        api = "https://datasets-server.huggingface.co/rows"
        while sum(map(len, texts)) < required_chars:
            query = urllib.parse.urlencode({
                "dataset": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1",
                "split": split,
                "offset": offset,
                "length": 100,
            })
            request = urllib.request.Request(
                f"{api}?{query}", headers={"User-Agent": "GrowingNN-Experiment012/1.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not fetch WikiText-2 {split} rows from Hugging Face. "
                    "Check network access or pass --text-file PATH"
                ) from exc
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                text = item.get("row", {}).get("text", "")
                if isinstance(text, str) and len(text.strip()) > 30:
                    texts.append(text.strip())
            offset += len(rows)
            total_rows = payload.get("num_rows_total", total_rows)
            print(f"  WikiText {split}: fetched {offset} rows, {sum(map(len, texts)):,} text chars",
                  flush=True)
            if total_rows is not None and offset >= total_rows:
                break
        if sum(map(len, texts)) < required_chars:
            raise RuntimeError(
                f"WikiText-2 {split} provided only {sum(map(len, texts)):,} usable characters; "
                f"need {required_chars:,}"
            )
        cache_file.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        return texts

    train = fetch_split("train", 100_000)
    validation = fetch_split("validation", 100_000)
    return train, validation, "Salesforce/wikitext:wikitext-2-raw-v1 via Hugging Face rows API"


def text_data(tokenizer, documents, count, seed):
    stream = []
    for text in documents:
        stream.extend(tokenizer.encode(text, add_special_tokens=False))
        stream.append(int(tokenizer.eos_token_id))
    indices = list(range(CONTEXT, len(stream) - 1))
    random.Random(seed).shuffle(indices)
    if len(indices) < count:
        raise ValueError(f"text split has {len(indices)} token positions; need {count}")
    indices = sorted(indices[:count])
    xs, ms, ys = [], [], []
    for i in indices:
        x, mask = pack_context(stream[max(0, i - CONTEXT):i], int(tokenizer.pad_token_id))
        xs.append(x)
        ms.append(mask)
        ys.append(stream[i])
    return tuple(torch.tensor(v, dtype=torch.long) for v in (xs, ms, ys))


def soft_targets(teacher, data, device, batch_size):
    ids, masks, _ = data
    all_ids, all_probs = [], []
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            x, mask = ids[start:start + batch_size].to(device), masks[start:start + batch_size].to(device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = teacher(input_ids=x, attention_mask=mask, position_ids=pos,
                             use_cache=False, return_dict=True).logits[:, -1].float()
            probabilities = logits.softmax(-1)
            probs, tokens = probabilities.topk(TOP_K, dim=-1)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
            all_ids.append(tokens.cpu())
            all_probs.append(probs.cpu())
    return torch.cat(all_ids), torch.cat(all_probs)


def loader(data, batch_size, shuffle=False, seed=SEED):
    x = pack_token_inputs(data[0], data[1])
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(x, data[2]), batch_size=batch_size,
                      shuffle=shuffle, generator=generator)


def train_soft(model, train_data, targets, val_loader, *, epochs, batch_size, lr, device, seed):
    x = pack_token_inputs(train_data[0], train_data[1])
    ids, probs = targets
    generator = torch.Generator().manual_seed(seed)
    data = DataLoader(TensorDataset(x, ids, probs), batch_size=batch_size,
                      shuffle=True, generator=generator)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total = 0
        for packed, token_ids, token_probs in data:
            packed = packed.to(device).float()
            token_ids, token_probs = token_ids.to(device), token_probs.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(packed).float()
            loss = -(token_probs * logits.log_softmax(-1).gather(1, token_ids)).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(packed)
            total_correct += int((logits.argmax(-1) == token_ids[:, 0]).sum())
            total += len(packed)
        val_loss, val_acc = _evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(total, 1),
                        "train_acc": total_correct / max(total, 1),
                        "val_loss": float(val_loss), "val_acc": float(val_acc)})
        print(f"  epoch {epoch + 1}/{epochs}: soft loss={history[-1]['train_loss']:.4f}; "
              f"held-out agreement={val_acc:.3f}", flush=True)
    return history


def count_params(model):
    seen, count = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            count += p.numel()
    return count


def save_hard(path: Path, data: tuple[torch.Tensor, ...]) -> int:
    """Persist a hard-label dataset and return its serialized size."""
    torch.save(
        {"input_ids": data[0], "attention_masks": data[1], "targets": data[2]},
        path,
    )
    return path.stat().st_size


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = resolve_device(args.device)
    epochs = args.epochs or (1 if args.smoke else 3)
    train_n, val_n, tokens_each = (64, 32, 4) if args.smoke else (8192, 2048, 16)
    if train_n % tokens_each or val_n % tokens_each:
        raise ValueError("target counts must be divisible by prompt continuation length")
    batch_size = args.batch_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = args.output_dir / "datasets"
    ds_dir.mkdir(exist_ok=True)
    print(f"Loading {args.model} on {device}; {'smoke' if args.smoke else 'medium'} run", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=args.local_files_only
    ).to(device).eval()

    n_train_prompts, n_val_prompts = train_n // tokens_each, val_n // tokens_each
    eval_set = set(EVAL_PROMPTS)
    pool = [t.format(topic=topic) for t in TEMPLATES for topic in TOPICS
            if t.format(topic=topic) not in eval_set]
    if n_train_prompts + n_val_prompts > len(pool):
        raise ValueError("expand the prompt pool for these requested target counts")
    shuffled = random.Random(SEED + 1).sample(pool, len(pool))
    train_prompts = shuffled[:n_train_prompts]
    val_prompts = shuffled[n_train_prompts:n_train_prompts + n_val_prompts]
    prompt_train = prompt_data(teacher, tokenizer, train_prompts, tokens_each, device, batch_size)
    prompt_val = prompt_data(teacher, tokenizer, val_prompts, tokens_each, device, batch_size)
    train_text, val_text, corpus_name = load_text(args.text_file, ds_dir)
    real_train = text_data(tokenizer, train_text, train_n, SEED + 2)
    real_val = text_data(tokenizer, val_text, val_n, SEED + 3)
    print("Building top-8 teacher targets on real-text contexts ...", flush=True)
    train_soft = soft_targets(teacher, real_train, device, batch_size)
    val_soft = soft_targets(teacher, real_val, device, batch_size)
    # Hard-label real-text examples use GPT-2's top-1 output, rather than the
    # corpus token, so this arm measures teacher-memory recovery.
    real_train_teacher = (real_train[0], real_train[1], train_soft[0][:, 0].clone())
    real_val_teacher = (real_val[0], real_val[1], val_soft[0][:, 0].clone())
    teacher_val = real_val_teacher

    half_train, half_val = train_n // 2, val_n // 2
    mixed_train = tuple(torch.cat((prompt_train[i][:half_train],
                                   real_train_teacher[i][:train_n-half_train]))
                        for i in range(3))
    mixed_val = tuple(torch.cat((prompt_val[i][:half_val],
                                 real_val_teacher[i][:val_n-half_val]))
                      for i in range(3))
    saved_bytes = {}
    for name, data in (("generated_prompts", prompt_train), ("real_text", real_train_teacher),
                       ("mixed", mixed_train)):
        saved_bytes[name] = save_hard(ds_dir / f"{name}_train.pt", data)
    soft_path = ds_dir / "soft_teacher_train.pt"
    torch.save({"input_ids": real_train[0], "attention_masks": real_train[1],
                "top_token_ids": train_soft[0], "top_probabilities": train_soft[1]}, soft_path)
    saved_bytes["soft_teacher"] = soft_path.stat().st_size
    for name, data in (("generated_prompts", prompt_val), ("real_text", real_val_teacher),
                       ("mixed", mixed_val)):
        save_hard(ds_dir / f"{name}_validation.pt", data)
    torch.save({"input_ids": real_val[0], "attention_masks": real_val[1],
                "top_token_ids": val_soft[0], "top_probabilities": val_soft[1]},
               ds_dir / "soft_teacher_validation.pt")
    (ds_dir / "prompt_splits.json").write_text(json.dumps(
        {"train": train_prompts, "validation": val_prompts}, indent=2), encoding="utf-8")

    sample = pack_token_inputs(real_train[0][:1], real_train[1][:1]).to(device)
    print("Capturing teacher with hf_symbolic_editable ...", flush=True)
    traced = TracedModel.create(CausalLMNextTokenAdapter(teacher), tuple(sample.shape),
                                capture="hf_symbolic_editable", example_args=(sample,))
    for path, module in list(traced.gm.named_modules()):
        if path and isinstance(module, nn.Linear) and module.out_features == len(tokenizer):
            traced.gm.set_submodule(path, FixedOutputProjection(module))
    traced.shapes()
    if not any(path == ACTION_TARGET for path, _ in traced.gm.named_modules()):
        raise RuntimeError(f"capture did not expose fixed action target {ACTION_TARGET}")
    prompt_examples = EVAL_PROMPTS
    teacher_generation = {
        "outputs": _hf_greedy_prompts(
            teacher, tokenizer, prompt_examples, context=1024, new_tokens=24, device=device
        )
    }
    # Use the teacher's top-1 prediction on held-out real-text contexts for the
    # common recovery metric.
    val_loader = loader(teacher_val, batch_size)
    report = {
        "experiment": "012_gpt2_memory_dataset_comparison", "status": "running",
        "model": args.model, "device": str(device), "seed": SEED, "smoke": args.smoke,
        "corpus": corpus_name, "train_targets": train_n, "validation_targets": val_n,
        "context_tokens": CONTEXT, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": 2e-5, "action_target": ACTION_TARGET, "action_ratio": ACTION_RATIO,
        "initial_parameters": traced.param_count(), "dataset_snapshot_bytes": saved_bytes,
        "eval_prompts": prompt_examples,
            "examples": {"teacher": {x["prompt"]: x["generated_text"] for x in teacher_generation["outputs"]}},
        "arms": {},
    }
    hard_arms = {"generated_prompts": prompt_train, "real_text": real_train_teacher,
                 "mixed": mixed_train}
    for i, name in enumerate(("generated_prompts", "real_text", "soft_teacher", "mixed")):
        print(f"\n=== {name} ({i+1}/4) ===", flush=True)
        started = time.perf_counter()
        arm_trace = copy.deepcopy(traced)
        DelNeurons([ACTION_TARGET, ACTION_RATIO]).execute(arm_trace)
        student = arm_trace.gm.to(device)
        action_loss, action_acc = _evaluate(student, val_loader, nn.CrossEntropyLoss(), device)
        action_only = {"loss": float(action_loss), "agreement": float(action_acc)}
        action_generation = _timed_greedy_prompts(
            student, tokenizer, prompt_examples, context=1024, new_tokens=24, device=device)
        action_output_map = {x["prompt"]: x["generated_text"] for x in action_generation["outputs"]}
        arm_dir = args.output_dir / "arms" / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        if name == "soft_teacher":
            history = train_soft(student, real_train, train_soft, val_loader,
                                 epochs=epochs, batch_size=batch_size, lr=2e-5,
                                 device=device, seed=SEED + i)
        else:
            train_loader = loader(hard_arms[name], batch_size, shuffle=True, seed=SEED + i)
            student, hist = gradient_descent(
                student, epochs, train_loader, val_loader, nn.CrossEntropyLoss(),
                ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=2e-5),
                device=device, quiet=False, print_every=1, max_grad_norm=1.0)
            history = [{"epoch": e+1, "train_loss": hist["train_loss"][e],
                        "train_acc": hist["train_acc"][e], "val_loss": hist["val_loss"][e],
                        "val_acc": hist["val_acc"][e]} for e in range(len(hist["train_loss"]))]
        final_loss, final_acc = _evaluate(student, val_loader, nn.CrossEntropyLoss(), device)
        generated = _timed_greedy_prompts(student, tokenizer, prompt_examples, context=1024,
                                         new_tokens=24, device=device)
        output_map = {x["prompt"]: x["generated_text"] for x in generated["outputs"]}
        report["examples"][name] = {"action_only": action_output_map, "trained": output_map}
        seconds = time.perf_counter() - started
        report["arms"][name] = {
            "snapshot_bytes": saved_bytes[name], "parameters": count_params(student),
            "action_only": action_only,
            "action_only_generation_agreement": _benchmark_output_agreement(
                teacher_generation, action_generation),
            "final": {"loss": float(final_loss), "agreement": float(final_acc)},
            "generation_agreement": _benchmark_output_agreement(teacher_generation, generated),
            "generation_seconds": generated["elapsed_seconds"], "seconds": seconds,
            "history": history,
        }
        torch.save({"state_dict": student.state_dict(), "parameter_count": count_params(student)},
                   arm_dir / "student_after_training.pt")
        (arm_dir / "history.json").write_text(json.dumps(report["arms"][name], indent=2),
                                               encoding="utf-8")
        del student, arm_trace
        if device.type == "cuda":
            torch.cuda.empty_cache()
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_report(report, args.output_dir)
    report["status"] = "smoke_completed" if args.smoke else "completed"
    report["total_training_seconds"] = sum(x["seconds"] for x in report["arms"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, args.output_dir)
    return report


def write_charts(report, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    arms = report.get("arms", {})
    if not arms:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, arm in arms.items():
        hist = arm["history"]
        epochs = [x["epoch"] for x in hist]
        axes[0].plot(epochs, [x["train_loss"] for x in hist], marker="o", label=f"{name} train")
        axes[0].plot(epochs, [x["val_loss"] for x in hist], marker=".", linestyle="--",
                     label=f"{name} held-out")
        axes[1].plot(epochs, [x["val_acc"] for x in hist], marker="o", label=name)
    axes[0].set_title("Hard-target loss on common held-out teacher labels")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Agreement with teacher on held-out real-text contexts")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 agreement")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(charts / "training_and_validation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(arms)
    values = [arms[name]["snapshot_bytes"] for name in names]
    ax.bar(names, values, color=["#2563eb", "#16a34a", "#9333ea", "#f97316"][:len(names)])
    ax.set_title("Serialized training snapshot size")
    ax.set_ylabel("Bytes")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(charts / "dataset_snapshot_sizes.png", dpi=150)
    plt.close(fig)
    return {"training_chart": "charts/training_and_validation.png",
            "dataset_size_chart": "charts/dataset_snapshot_sizes.png"}


def write_report(report, out):
    chart_paths = write_charts(report, out)
    rows = [
        "# Experiment 012: GPT-2 memory dataset comparison", "",
        f"- Status: {report['status']}", f"- Model: {report['model']}",
        f"- Device: {report['device']}", f"- Corpus: {report['corpus']}",
        f"- Target positions per arm: {report['train_targets']:,} train / {report['validation_targets']:,} validation",
        f"- Fixed action: {ACTION_TARGET}, ratio {ACTION_RATIO}", "",
        "| Arm | Snapshot bytes | Parameters | Action-only agreement | Final agreement | Generation agreement | Final loss | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in report.get("arms", {}).items():
        rows.append(f"| {name} | {arm['snapshot_bytes']:,} | {arm['parameters']:,} | "
                    f"{arm['action_only']['agreement']:.1%} | {arm['final']['agreement']:.1%} | "
                    f"{arm['generation_agreement']['token_accuracy']:.1%} | "
                    f"{arm['final']['loss']:.4f} | {arm['seconds']:.1f} |")
    if chart_paths:
        rows += ["", f"![Training and held-out history]({chart_paths['training_chart']})", "",
                 f"![Dataset snapshot sizes]({chart_paths['dataset_size_chart']})", ""]
    for name, arm in report.get("arms", {}).items():
        rows += ["", f"## {name}", "",
                 "| Epoch | Train loss | Train agreement | Held-out loss | Held-out agreement |",
                 "|---:|---:|---:|---:|---:|"]
        for entry in arm["history"]:
            rows.append(f"| {entry['epoch']} | {entry['train_loss']:.4f} | "
                        f"{entry['train_acc']:.1%} | {entry['val_loss']:.4f} | {entry['val_acc']:.1%} |")
    rows += ["", "## Prompt examples", ""]
    if report.get("examples"):
        for prompt in report["eval_prompts"]:
            rows += [f"### {prompt}", "", f"**Teacher:** {report['examples']['teacher'][prompt]}", "",
                     "| Dataset arm | After shrink, before training | After training |",
                     "|---|---|---|"]
            for name in report["arms"]:
                action_output = report["examples"][name]["action_only"][prompt]
                trained_output = report["examples"][name]["trained"][prompt]
                action_output = action_output.replace("|", "\\|").replace("\n", " ")
                trained_output = trained_output.replace("|", "\\|").replace("\n", " ")
                rows.append(f"| {name} | {action_output} | {trained_output} |")
    rows += ["", "The soft-target arm uses an experiment-local top-8 distillation loss. "
             "Held-out text and free-running generations are both needed to judge recovery.", ""]
    (out / "report.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    args = parse_args()
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    result = run(args)
    print(f"Experiment 012 {result['status']}; report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
