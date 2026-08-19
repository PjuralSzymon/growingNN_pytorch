"""
Regression: HuggingFace FX trace of tiny GPT-2, then GrowingNN actions on copies.

Prints live-model and traced-graph module types into graph_summary0.txt.

Run:
  python tests/regression/actions/transformer_all_action_test.py
"""

import copy
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_extraction import extract_graph
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.regression.regression_utils import (
    FOLDER_NAME,
    log_action_count_table,
    log_regression_action_error,
    parse_regression_cli,
)

MODEL_ID = "sshleifer/tiny-gpt2"
OUT_DIR = FOLDER_NAME + "/transformer"
LOOK_FOR = ("c_attn", "c_proj", "c_fc", "q_proj", "k_proj", "v_proj")


if __name__ == "__main__":
    parse_regression_cli()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    try:
        import transformers
        from transformers import AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        logger.error("need transformers 4.x with utils.fx: %s", exc)
        sys.exit(1)
    logger.info("transformers %s", transformers.__version__)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    model.config.use_cache = False
    n_embd = int(model.config.n_embd)
    x = torch.randn(2, 8, n_embd)
    trace_shape = (1, 8, n_embd)
    logger.info("loaded %s n_embd=%s x=%s", MODEL_ID, n_embd, tuple(x.shape))

    try:
        gm = extract_graph(model, input_names=["inputs_embeds"])
    except Exception as exc:
        logger.error("extract_graph failed: %s: %s", type(exc).__name__, exc)
        sys.exit(1)
    with torch.no_grad():
        out = gm(x)
    logger.info("extract_graph forward ok out type=%s", type(out).__name__)

    traced = TracedModel.create(gm, trace_shape)
    try:
        out_shapes, in_shapes = traced.shapes()
    except Exception as exc:
        out_shapes, in_shapes = {}, {}
        logger.info("shapes failed: %s: %s", type(exc).__name__, exc)

    lines = []
    lines.append("=== types: original model named_modules ===")
    lines.append("name | class | module | extra")
    type_counts = Counter()
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        pkg = type(mod).__module__
        extra = ""
        if isinstance(mod, nn.Linear):
            extra = f"Linear({mod.in_features}, {mod.out_features})"
        elif hasattr(mod, "nf") and hasattr(mod, "weight"):
            extra = f"nf={mod.nf} weight={tuple(mod.weight.shape)}"
        elif hasattr(mod, "weight") and getattr(mod.weight, "shape", None) is not None:
            extra = f"weight={tuple(mod.weight.shape)}"
        type_counts[f"{pkg}.{cls}"] += 1
        flag = " LOOK" if any(key in name for key in LOOK_FOR) else ""
        lines.append(f"{name or '(root)'} | {cls} | {pkg} | {extra}{flag}")
        logger.info("orig %s | %s | %s | %s%s", name or "(root)", cls, pkg, extra, flag)
    lines.append("")
    lines.append("original type counts:")
    for key, n in sorted(type_counts.items()):
        lines.append(f"{key}: {n}")

    lines.append("")
    lines.append("=== types: traced graph call_module ===")
    lines.append("target | class | module | extra | in_shape | out_shape")
    traced_type_counts = Counter()
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        try:
            mod = gm.get_submodule(str(node.target))
        except AttributeError:
            lines.append(f"{node.target} | unresolved | | | |")
            continue
        cls = type(mod).__name__
        pkg = type(mod).__module__
        extra = ""
        if isinstance(mod, nn.Linear):
            extra = f"Linear({mod.in_features}, {mod.out_features})"
        elif hasattr(mod, "nf") and hasattr(mod, "weight"):
            extra = f"nf={mod.nf} weight={tuple(mod.weight.shape)}"
        elif hasattr(mod, "weight") and getattr(mod.weight, "shape", None) is not None:
            extra = f"weight={tuple(mod.weight.shape)}"
        traced_type_counts[f"{pkg}.{cls}"] += 1
        flag = " LOOK" if any(key in str(node.target) for key in LOOK_FOR) else ""
        s_in = in_shapes.get(str(node.target))
        s_out = out_shapes.get(str(node.target))
        lines.append(f"{node.target} | {cls} | {pkg} | {extra} | {s_in} | {s_out}{flag}")
        logger.info(
            "traced %s | %s | %s | %s | in=%s out=%s%s",
            node.target, cls, pkg, extra, s_in, s_out, flag,
        )
    lines.append("")
    lines.append("traced call_module type counts:")
    for key, n in sorted(traced_type_counts.items()):
        lines.append(f"{key}: {n}")

    lines.append("")
    lines.append("=== get_attr weight targets (inlined modules) ===")
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        lines.append(str(node.target))

    cfg = RunningConfig(generations=1, epochs=1)
    cfg.update_grow_actions(True)
    cfg.update_shrink_actions(True)
    seq_actions = AddSeqLinearLayer.generate_all_actions(traced)
    res_actions = AddResLinearLayer.generate_all_actions(traced)
    all_actions = generate_all_actions(traced, cfg)
    counts = Counter(type(a).__name__ for a in all_actions)

    lines.append("")
    lines.append("=== GrowingNN pairs and Linear actions ===")
    lines.append(f"hidden_modules: {traced.hidden_modules()}")
    lines.append(f"sequential_pairs: {traced.sequential_pairs()}")
    lines.append(f"dependency_pairs: {traced.dependency_pairs()}")
    lines.append(f"AddSeqLinearLayer.generate_all_actions: {len(seq_actions)}")
    lines.append(f"AddResLinearLayer.generate_all_actions: {len(res_actions)}")
    logger.info("sequential_pairs: %s", traced.sequential_pairs())
    logger.info("dependency_pairs: %s", traced.dependency_pairs())
    lines.extend([""] + log_action_count_table(
        {"AddSeqLinearLayer": len(seq_actions), "AddResLinearLayer": len(res_actions)},
        title="linear generators (before registry filter)",
        include_known_zeros=False,
    ))
    lines.extend([""] + log_action_count_table(dict(counts), title="registry generate_all_actions"))
    for name in ("AddSeqLinearLayer", "AddResLinearLayer"):
        logger.info("%s | %s", name, counts.get(name, 0))
        for i, action in enumerate(a for a in all_actions if type(a).__name__ == name):
            logger.info("%s[%s]: %s", name, i, action)

    lines.append("")
    lines.append("=== graph ===")
    lines.append(f"model={MODEL_ID}")
    lines.append(f"input_shape={trace_shape}")
    lines.append(f"opcodes={dict(Counter(node.op for node in gm.graph.nodes))}")
    lines.append("")
    lines.append("nodes: name | op | target | class | extra | in_shape | out_shape | inputs | users")
    for node in gm.graph.nodes:
        cls = extra = ""
        s_in = s_out = ""
        if node.op == "call_module":
            try:
                mod = gm.get_submodule(str(node.target))
                cls = type(mod).__name__
                if isinstance(mod, nn.Linear):
                    extra = f"Linear({mod.in_features}, {mod.out_features})"
                elif hasattr(mod, "weight") and getattr(mod.weight, "shape", None) is not None:
                    extra = f"weight={tuple(mod.weight.shape)}"
            except AttributeError:
                cls = "unresolved"
            s_in = in_shapes.get(str(node.target), "")
            s_out = out_shapes.get(str(node.target), "")
        inputs = ",".join(inp.name for inp in node.all_input_nodes)
        users = ",".join(user.name for user in node.users)
        lines.append(
            f"{node.name} | {node.op} | {node.target} | {cls} | {extra} | {s_in} | {s_out} | {inputs} | {users}"
        )
    lines.append("")
    lines.append("raw graph:")
    lines.append(str(gm.graph))
    lines.append("")
    lines.append("generated code:")
    lines.append(gm.code)

    summary_path = Path(OUT_DIR) / "graph_summary0.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("saved %s", summary_path)

    draw_filtered_fx_graph(gm, OUT_DIR + "/fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, OUT_DIR + "/fx_graph0", fmt="pdf")

    logger.info("generated %s actions", len(all_actions))
    log_action_count_table(dict(counts), title="executing these actions")
    if not all_actions:
        logger.warning("no GrowingNN actions generated")
        sys.exit(0)

    executed = Counter()
    for idx, action in enumerate(all_actions):
        name = type(action).__name__
        gm_copy = copy.deepcopy(gm)
        traced_copy = TracedModel.create(gm_copy, trace_shape)
        logger.info("execute %s/%s type=%s action=%s", idx + 1, len(all_actions), name, action)
        try:
            action.execute(traced_copy)
            with torch.no_grad():
                traced_copy.gm(x)
        except Exception:
            draw_filtered_fx_graph(
                traced_copy.gm, OUT_DIR + f"/fx_graph_simplified_error_{idx + 1}_{name}", fmt="pdf"
            )
            log_regression_action_error(
                traced_copy.gm, action, actions=all_actions, idx=idx, action_type=name
            )
            log_action_count_table(dict(executed), title="executed ok before failure")
            log_action_count_table(dict(counts), title="generated (failure remaining)")
            raise
        executed[name] += 1
        draw_filtered_fx_graph(traced_copy.gm, OUT_DIR + f"/fx_graph_simplified{idx + 1}_{name}", fmt="pdf")
        draw_torch_fx_graph(traced_copy.gm, OUT_DIR + f"/fx_graph{idx + 1}_{name}", fmt="pdf")
    log_action_count_table(dict(executed), title="executed ok")
    logger.info("done.")
