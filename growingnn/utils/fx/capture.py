"""Capture adapters that normalize supported routes to ``torch.fx.GraphModule``."""

from __future__ import annotations

import inspect

import torch
import torch.fx as fx
import torch.nn as nn


def order_example_inputs(
    gm: fx.GraphModule,
    example_args: tuple[object, ...],
    example_kwargs: dict[str, object] | None = None,
) -> tuple[object, ...]:
    """Arrange positional and named samples in the captured placeholder order."""
    named = dict(example_kwargs or {})
    positional = iter(example_args)
    ordered: list[object] = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if node.name in named:
            ordered.append(named.pop(node.name))
            continue
        if node.name == "kwargs" and named:
            ordered.append(named)
            named = {}
            continue
        try:
            ordered.append(next(positional))
        except StopIteration:
            if node.args:
                ordered.append(node.args[0])
            else:
                raise ValueError(
                    f"No example value supplied for FX placeholder {node.name!r}"
                ) from None
    try:
        next(positional)
    except StopIteration:
        pass
    else:
        raise ValueError("Too many positional example inputs for captured FX placeholders")
    if named:
        raise ValueError(
            f"Example keyword inputs do not match graph placeholders: {tuple(named)}"
        )
    return tuple(ordered)


def _dummy_inputs(
    model: nn.Module,
    example_args: tuple[object, ...],
    example_kwargs: dict[str, object] | None,
) -> dict[str, object]:
    """Bind samples by the model signature for Hugging Face's named tracer inputs."""
    try:
        bound = inspect.signature(model.forward).bind_partial(
            *example_args, **(example_kwargs or {})
        )
    except TypeError as exc:
        raise ValueError(f"Example inputs do not match model.forward: {exc}") from exc
    return dict(bound.arguments)


def _capture_hf_editable_linear_graph(
    model: nn.Module,
    example_args: tuple[object, ...],
    example_kwargs: dict[str, object] | None = None,
) -> fx.GraphModule:
    """Capture HF Conv1D projections as native-rank editable Linear nodes.

    PyTorch Linear already applies over the final feature axis for arbitrary
    leading dimensions. Preserve that native interface instead of inserting
    flatten/reshape nodes with captured constant widths; keeping the actual
    rank-3 tensors visible is necessary for general feature-axis analysis and
    coordinated projection-width edits.
    """
    try:
        from transformers.pytorch_utils import Conv1D
        from transformers.utils.fx import HFTracer
    except ImportError as exc:
        raise RuntimeError("capture route 'hf_symbolic_editable' requires Transformers HFTracer and Conv1D") from exc

    class ProjectionLeafTracer(HFTracer):
        def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
            # Preserve embedding lookups as functional graph operations instead of
            # making learned token/position tables appear as generic removable layers.
            if isinstance(module, nn.Embedding):
                return False
            # Treat every Hugging Face Conv1D projection uniformly. Candidate
            # discovery later selects graph seams from topology and interfaces;
            # restricting capture to a path suffix hides attention projections
            # and ties visibility to one model's naming convention.
            if isinstance(module, Conv1D):
                return not self._stateless_mod_instanciation_depends_on_proxies(module)
            return super().is_leaf_module(module, module_qualified_name)

    dummy_inputs = _dummy_inputs(model, example_args, example_kwargs)
    tracer = ProjectionLeafTracer()
    graph = tracer.trace(model, dummy_inputs=dummy_inputs)
    gm = fx.GraphModule(model, graph)

    # Convert Hugging Face Conv1D parameter orientation to an actual nn.Linear.
    # This is a capture-time representation adapter; the source model is untouched.
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        module = gm.get_submodule(str(node.target))
        if not isinstance(module, Conv1D):
            continue
        linear = nn.Linear(module.weight.shape[0], module.weight.shape[1], bias=module.bias is not None)
        linear = linear.to(device=module.weight.device, dtype=module.weight.dtype)
        linear.train(module.training)
        with torch.no_grad():
            linear.weight.copy_(module.weight.transpose(0, 1))
            if module.bias is not None:
                linear.bias.copy_(module.bias)
        gm.set_submodule(str(node.target), linear)
    return gm


class ConfigurableTracer(fx.Tracer):
    """Symbolic tracer with caller-controlled module leaf policy."""

    def __init__(self, *, leaf_module_types: tuple[type[nn.Module], ...] = ()):
        super().__init__()
        self.leaf_module_types = leaf_module_types

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if self.leaf_module_types and isinstance(m, self.leaf_module_types):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def capture_graph(
    model: nn.Module,
    *,
    route: str = "symbolic",
    example_args: tuple[object, ...] | None = None,
    example_kwargs: dict[str, object] | None = None,
    leaf_module_types: tuple[type[nn.Module], ...] = (),
) -> fx.GraphModule:
    """Capture as FX via legacy symbolic, HF metadata-aware symbolic, or torch.export."""
    if route == "symbolic":
        if not leaf_module_types:
            return fx.symbolic_trace(model)
        tracer = ConfigurableTracer(leaf_module_types=leaf_module_types)
        graph = tracer.trace(model)
        return fx.GraphModule(model, graph)
    if route == "export":
        if example_args is None and not example_kwargs:
            raise ValueError("capture route 'export' requires example_args or example_kwargs")
        exported = torch.export.export(model, example_args or (), kwargs=example_kwargs)
        exported_module = exported.module()
        # ExportedProgram's generated module overrides train/eval as unsupported.
        # Re-root its already-captured graph as a regular FX GraphModule so the
        # shared ShapeProp and graph-editor path can be used by downstream code.
        return fx.GraphModule(exported_module, exported_module.graph)
    if route in ("hf_symbolic", "hf_symbolic_editable"):
        if example_args is None and not example_kwargs:
            raise ValueError(f"capture route {route!r} requires example_args or example_kwargs")
        example_args = example_args or ()
        if route == "hf_symbolic_editable":
            return _capture_hf_editable_linear_graph(model, example_args, example_kwargs)
        try:
            from transformers.utils.fx import HFTracer
        except ImportError as exc:
            raise RuntimeError(
                "capture route 'hf_symbolic' requires a Transformers version that provides transformers.utils.fx.HFTracer"
            ) from exc
        dummy_inputs = _dummy_inputs(model, example_args, example_kwargs)
        tracer = HFTracer()
        graph = tracer.trace(model, dummy_inputs=dummy_inputs)
        return fx.GraphModule(model, graph)
    raise ValueError(
        f"Unknown FX capture route: {route!r}. Expected 'symbolic', 'hf_symbolic', 'hf_symbolic_editable', or 'export'."
    )
