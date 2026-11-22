"""Discover residual regions in a small ResNet and color each region in a graph image.

Run from the repository root with the project's Python environment. The script
accepts any importable ``nn.Module`` factory by editing ``_make_model``.
"""

from __future__ import annotations

from copy import deepcopy
import sys
import re
from pathlib import Path

import torch
import torch.fx as fx
from graphviz import Digraph
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from growingnn.actions.residual_block import ResidualBlockAction, discover_residual_regions
from growingnn.core.traced_model import TracedModel


def _make_model():
    # Official Torchvision ImageNet checkpoint. CIFAR-10 is used below as an
    # easy real-image probe, resized by the checkpoint's recommended transform.
    weights = ResNet18_Weights.DEFAULT
    return resnet18(weights=weights).eval(), weights, (1, 3, 224, 224)


def _make_probe(weights, data_dir: Path):
    dataset = CIFAR10(root=str(data_dir), train=False, download=True)
    image, label = dataset[0]
    preprocess = weights.transforms()
    tensor = preprocess(image).unsqueeze(0)
    return tensor, dataset.classes[label]


def _save_prediction_report(path: Path, variants: dict[str, torch.Tensor], class_name: str) -> None:
    reference = variants["00_original"]
    lines = [f"CIFAR-10 sample label: {class_name}", "Logits from pretrained ImageNet ResNet-18:"]
    for name, logits in variants.items():
        probs = logits.softmax(dim=1)
        score, index = probs.max(dim=1)
        delta = (logits - reference).abs().max().item()
        lines.append(
            f"{name}: top1={ResNet18_Weights.DEFAULT.meta['categories'][index.item()]} "
            f"({score.item():.5f}), max_abs_logit_delta={delta:.6g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw(gm: fx.GraphModule, regions, output: Path, *, color_by_region=None) -> None:
    dot = Digraph("residual_regions")
    dot.attr(rankdir="LR", compound="true", label="FX graph: discovered residual regions")
    palette = ["#ffd6a5", "#caffbf", "#9bf6ff", "#bdb2ff", "#ffc6ff", "#fdffb6", "#a0c4ff", "#ffadad"]
    color_by_region = color_by_region or {}
    membership: dict[fx.Node, list[int]] = {}
    for region_index, (_, _, members, reason) in enumerate(regions):
        if reason:
            continue
        for node in members:
            membership.setdefault(node, []).append(region_index)

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            label = f"{node.op}: {node.name}"
        elif node.op == "call_module":
            label = f"{node.target}\n{node.name}"
        else:
            label = f"{node.op}: {getattr(node.target, '__name__', node.target)}\n{node.name}"
        regions_here = membership.get(node, [])
        active_region = regions_here[-1] if regions_here else None
        fillcolor = color_by_region.get(active_region, palette[active_region % len(palette)] if active_region is not None else "white")
        attrs = {"label": label, "style": "filled", "fillcolor": fillcolor}
        if regions_here:
            attrs["tooltip"] = "candidate blocks: " + ", ".join(str(index + 1) for index in regions_here)
        dot.node(node.name, **attrs)
        for parent in node.all_input_nodes:
            dot.edge(parent.name, node.name)

    output.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(output), format="png", cleanup=True)


def _module_signature(gm, region):
    _, _, members, _ = region
    return {
        re.sub(r"_block_\d+$", "", str(node.target))
        for node in members
        if node.op == "call_module"
    }


def _inherit_colors(gm, regions, original_regions):
    original_signatures = [_module_signature(gm, region) for region in original_regions]
    mapping = {}
    for index, region in enumerate(regions):
        signature = _module_signature(gm, region)
        if not signature:
            continue
        overlaps = [len(signature & original) for original in original_signatures]
        if overlaps and max(overlaps) > 0:
            mapping[index] = "#caffbf" if overlaps.index(max(overlaps)) == 0 else "#bdb2ff"
    return mapping


def main() -> None:
    model, weights, input_shape = _make_model()
    probe, class_name = _make_probe(weights, ROOT / "testResults" / "regression_cache" / "cifar10")
    gm = fx.symbolic_trace(model)
    traced = TracedModel.create(gm, input_shape)
    regions = discover_residual_regions(traced)
    supported = [region for region in regions if region[3] is None]
    print(f"Residual add candidates: {len(regions)}")
    print(f"Shape-compatible regions: {len(supported)}")
    for index, (entry, exit_node, members, reason) in enumerate(regions, start=1):
        status = "supported" if reason is None else f"rejected: {reason}"
        print(f"{index:02d}. {entry.name} -> {exit_node.name} ({len(members)} FX nodes): {status}")
    image_dir = ROOT / "testResults" / "regression" / "residual_block_variants"
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "#caffbf", 1: "#bdb2ff"}  # first supported = green; second = purple
    _draw(gm, regions, image_dir / "00_original", color_by_region=colors)
    predictions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        predictions["00_original"] = gm(probe).cpu()

    # Keep the chosen targets fixed from the original graph. Every variant is
    # made from a fresh GraphModule copy so mutations cannot leak between views.
    chosen = [("remove_green", 0, "remove"), ("duplicate_purple", 1, "duplicate"),
              ("remove_purple", 1, "remove"), ("duplicate_green", 0, "duplicate")]
    for label, region_index, operation in chosen:
        if region_index >= len(supported):
            print(f"Skipping {label}: need at least {region_index + 1} supported regions")
            continue
        source_entry, source_exit, _, _ = supported[region_index]
        variant_gm = deepcopy(gm)
        variant = TracedModel.create(variant_gm, input_shape)
        matches = [
            action for action in ResidualBlockAction.generate_all_actions(variant)
            if action.params[0] == operation
            and action.params[1] == source_entry.name
            and action.params[2] == source_exit.name
        ]
        if not matches:
            print(f"Skipping {label}: selected action was not generated")
            continue
        matches[0].execute(variant)
        with torch.no_grad():
            predictions[label] = variant_gm(probe).cpu()
        changed_regions = discover_residual_regions(variant)
        inherited_colors = _inherit_colors(variant_gm, changed_regions, supported)
        _draw(variant_gm, changed_regions, image_dir / label, color_by_region=inherited_colors)
        print(f"Saved {label}: {image_dir / (label + '.png')}")
    _save_prediction_report(image_dir / "output_comparison.txt", predictions, class_name)
    print(f"Saved output comparison: {image_dir / 'output_comparison.txt'}")
    print(f"Saved original and mutation views under {image_dir}")


if __name__ == "__main__":
    main()
