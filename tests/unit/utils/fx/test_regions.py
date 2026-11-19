import operator
from collections import UserDict, namedtuple

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from growingnn.utils.fx.regions import GraphRegionAnalyzer


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        a = torch.relu(self.fc1(x))
        return a + self.fc2(a)


def test_regions_keep_functional_ops_and_nested_boundaries():
    gm = fx.symbolic_trace(_Residual())
    ShapeProp(gm).propagate(torch.randn(2, 4))
    candidates = GraphRegionAnalyzer.discover(gm)
    assert candidates
    assert any(node.op == "call_function" and node.target is torch.relu
               for candidate in candidates for node in candidate.nodes)
    assert any(len(candidate.nodes) > 1 for candidate in candidates)
    assert all(candidate.entry in candidate.nodes and candidate.exit in candidate.nodes
               for candidate in candidates)
    assert any(candidate.input_interface and candidate.output_interface for candidate in candidates)
    tensor_interfaces = [
        interface
        for candidate in candidates
        for interface in (*candidate.input_interface, *candidate.output_interface)
        if interface.kind == "tensor"
    ]
    assert tensor_interfaces
    assert all(interface.shape is not None and interface.dtype is not None for interface in tensor_interfaces)


def test_postdominating_boundary_summary_handles_deep_module_chains():
    from growingnn.utils.fx.regions import _dominator_tree, _nearest_postdominating_boundaries

    model = nn.Sequential(*(nn.Linear(4, 4) for _ in range(256))).eval()
    gm = fx.symbolic_trace(model)
    nodes = list(gm.graph.nodes)
    placeholders = {node for node in nodes if node.op == "placeholder"}
    outputs = {node for node in nodes if node.op == "output"}
    parent, _ = _dominator_tree(nodes, outputs, reverse=True)

    nearest_modules, nearest_merges = _nearest_postdominating_boundaries(nodes, parent)
    module_nodes = [node for node in nodes if node.op == "call_module"]

    assert nearest_modules[module_nodes[0]] is module_nodes[1]
    assert nearest_modules[module_nodes[-1]] is None
    assert nearest_merges[module_nodes[0]] == ()

    # The full discovery route consumes the same summaries and keeps adjacent
    # module boundaries available without enumerating every possible pair.
    candidates = GraphRegionAnalyzer.discover(gm)
    assert any(
        candidate.entry is module_nodes[0] and candidate.exit is module_nodes[1]
        for candidate in candidates
    )


def test_pair_traversal_policy_keeps_small_graphs_and_bounds_large_ones():
    from growingnn.utils.fx import pair_traversal_decision

    small = fx.symbolic_trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)))
    small_decision = pair_traversal_decision(small)
    assert small_decision.use_legacy_pairs
    assert not small_decision.reasons

    large = fx.symbolic_trace(nn.Sequential(*(nn.ReLU() for _ in range(1000))))
    large_decision = pair_traversal_decision(large)
    assert not large_decision.use_legacy_pairs
    assert large_decision.estimated_walk_steps > 50_000
    assert any("exceeds budget" in reason for reason in large_decision.reasons)


def test_region_interfaces_preserve_structured_fx_values():
    class StructuredOutput(nn.Module):
        def forward(self, x):
            return torch.max(x, dim=-1)

    gm = fx.symbolic_trace(StructuredOutput())
    ShapeProp(gm).propagate(torch.randn(2, 3, 4))
    candidates = GraphRegionAnalyzer.discover(gm)
    max_node = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is torch.max)
    candidate = next(candidate for candidate in candidates if candidate.entry is max_node and candidate.exit is max_node)
    boundary = candidate.output_interface[0]
    assert boundary.kind not in ("unsupported", "tensor")
    assert len(boundary.children) == 2
    assert all(child.kind == "tensor" for child in boundary.children)


def test_value_interface_preserves_custom_mapping_and_namedtuple_types():
    class Identity(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    gm = fx.symbolic_trace(Identity())
    node = next(node for node in gm.graph.nodes if node.op == "call_function")
    pair_type = namedtuple("Pair", "left optional count")
    leaf = torch.empty(2, 3)
    node.meta["val"] = UserDict({0: leaf, "0": pair_type(leaf, None, 3)})

    interface = GraphRegionAnalyzer.describe_node_value(node)

    assert interface.kind == "mapping"
    assert interface.type_name == "UserDict"
    assert interface.keys == (0, "0")
    assert interface.children[0].kind == "tensor"
    assert interface.children[1].kind == "tuple"
    assert interface.children[1].type_name == "Pair"
    assert interface.children[1].children[1].kind == "none"
    assert interface.children[1].children[2].kind == "scalar"
    assert interface.children[1].children[2].constant == 3


def test_region_reports_unsupported_values_nested_in_structured_boundaries():
    class Identity(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    gm = fx.symbolic_trace(Identity())
    ShapeProp(gm).propagate(torch.randn(2, 3))
    node = next(node for node in gm.graph.nodes if node.op == "call_function")
    node.meta["val"] = UserDict({"tensor": torch.empty(2, 3), "custom": object()})

    candidate = next(
        region for region in GraphRegionAnalyzer.discover(gm)
        if region.entry is node and region.exit is node
    )

    assert not candidate.interface_analyzable
    assert any("['custom']" in reason and "unsupported type object" in reason
               for reason in candidate.interface_reasons)


def test_region_reports_missing_metadata_without_confusing_it_with_none_values():
    class Identity(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    gm = fx.symbolic_trace(Identity())
    relu = next(node for node in gm.graph.nodes if node.op == "call_function")
    candidate = next(
        region for region in GraphRegionAnalyzer.discover(gm)
        if region.entry is relu and region.exit is relu
    )

    assert not candidate.interface_analyzable
    assert any("metadata is missing" in reason for reason in candidate.interface_reasons)

    placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
    placeholder.meta["val"] = torch.empty(2, 3)
    relu.meta["val"] = (torch.empty(2, 3), None)
    candidate = next(
        region for region in GraphRegionAnalyzer.discover(gm)
        if region.entry is relu and region.exit is relu
    )
    assert candidate.interface_analyzable
    assert candidate.output_interface[0].children[1].kind == "none"


def test_action_sites_include_functional_nodes_and_dtype_rejection_reasons():
    gm = fx.symbolic_trace(_Residual())
    ShapeProp(gm).propagate(torch.randn(2, 4))
    sites = GraphRegionAnalyzer.discover_action_sites(gm)
    assert any(site.edge.source.op == "call_function" and site.interface.kind == "tensor" for site in sites)
    assert any(site.meaningful_seam for site in sites)
    assert all(
        site.edge.destination.op != "output" for site in sites if site.meaningful_seam
    )

    class IntegerAdd(nn.Module):
        def forward(self, x):
            return x + 1

    int_gm = fx.symbolic_trace(IntegerAdd())
    ShapeProp(int_gm).propagate(torch.ones(2, 4, dtype=torch.int64))
    int_sites = GraphRegionAnalyzer.discover_action_sites(int_gm)
    assert int_sites
    assert any(not site.eligible and any("dtype" in reason for reason in site.rejection_reasons)
               for site in int_sites)

    class MetadataConsumer(nn.Module):
        def forward(self, x):
            return x + x.size(0)

    metadata_gm = fx.symbolic_trace(MetadataConsumer())
    ShapeProp(metadata_gm).propagate(torch.randn(2, 4))
    metadata_sites = GraphRegionAnalyzer.discover_action_sites(metadata_gm)
    assert any(
        not site.eligible and any("tensor metadata" in reason for reason in site.rejection_reasons)
        for site in metadata_sites
    )


def test_region_boundary_records_repeated_fx_operand_occurrences():
    class RepeatedUse(nn.Module):
        def forward(self, x):
            return x + x

    gm = fx.symbolic_trace(RepeatedUse())
    ShapeProp(gm).propagate(torch.randn(2, 4))
    add = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is operator.add)
    candidate = next(
        region for region in GraphRegionAnalyzer.discover(gm)
        if region.entry is add and region.exit is add
    )
    boundary = next(edge for edge in candidate.inputs if edge.source.op == "placeholder")
    assert boundary.use_paths == (("args", 0), ("args", 1))


def test_region_boundary_records_fx_slice_operand_occurrences():
    class SliceModel(nn.Module):
        def forward(self, x, index):
            return x[index:index]

    gm = fx.symbolic_trace(SliceModel())
    ShapeProp(gm).propagate(torch.arange(8), 2)
    index = next(node for node in gm.graph.nodes if node.op == "placeholder" and node.name == "index")
    getitem = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is operator.getitem)
    region = next(
        region for region in GraphRegionAnalyzer.discover(gm)
        if region.entry is getitem and region.exit is getitem
    )
    edge = next(edge for edge in region.inputs if edge.source is index)

    assert edge.use_paths == (("args", 1, "start"), ("args", 1, "stop"))


def test_export_capture_requires_example_args_and_produces_graph_module():
    from growingnn.utils.fx.capture import capture_graph

    model = nn.Linear(4, 3)
    gm = capture_graph(model, route="export", example_args=(torch.randn(2, 4),))
    assert isinstance(gm, fx.GraphModule)


def test_linear_operation_descriptors_cover_module_and_functional_export_forms():
    from growingnn.utils.fx import capture_graph, describe_linear_operations

    model = nn.Sequential(nn.Linear(4, 7), nn.ReLU(), nn.Linear(7, 3)).eval()
    example = torch.randn(2, 5, 4)
    module_graph = fx.symbolic_trace(model)
    module_descriptors = describe_linear_operations(module_graph)
    assert [(item.in_features, item.out_features, item.form) for item in module_descriptors] == [
        (4, 7, "module"), (7, 3, "module")
    ]
    assert all(item.resize_support == "module_linear_rule" for item in module_descriptors)

    exported_graph = capture_graph(model, route="export", example_args=(example,))
    functional_descriptors = describe_linear_operations(exported_graph)
    assert [(item.in_features, item.out_features, item.form) for item in functional_descriptors] == [
        (4, 7, "function"), (7, 3, "function")
    ]
    assert all(item.recognized for item in functional_descriptors)
    assert all(item.weight_node.op == "get_attr" for item in functional_descriptors)
    assert all(item.parameter_aliases for item in functional_descriptors)
    assert all(item.resize_support == "requires_chain_analysis" for item in functional_descriptors)
    assert all("depends on its graph path" in item.diagnostic for item in functional_descriptors)

    class FunctionalLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 4))
            self.bias = nn.Parameter(torch.randn(5))

        def forward(self, value):
            return nn.functional.linear(value, self.weight, self.bias)

    symbolic_graph = fx.symbolic_trace(FunctionalLinear())
    symbolic_descriptors = describe_linear_operations(symbolic_graph)
    assert len(symbolic_descriptors) == 1
    assert symbolic_descriptors[0].form == "function"
    assert symbolic_descriptors[0].recognized
    assert (symbolic_descriptors[0].in_features, symbolic_descriptors[0].out_features) == (4, 5)


def test_linear_descriptors_resolve_addmm_weight_orientation_and_parameters():
    class AddmmProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 4))
            self.bias = nn.Parameter(torch.randn(5))

        def forward(self, value):
            return torch.addmm(self.bias, value, self.weight.t())

    from growingnn.utils.fx import describe_linear_operations

    model = AddmmProjection().eval()
    gm = fx.symbolic_trace(model)
    example = torch.randn(3, 4)
    descriptor = next(
        item for item in describe_linear_operations(gm) if item.form == "addmm"
    )

    assert descriptor.recognized
    assert (descriptor.in_features, descriptor.out_features) == (4, 5)
    assert descriptor.weight_target == "weight"
    assert descriptor.parameter_aliases == ("weight",)
    assert descriptor.weight_orientation == "out_features_by_in_features"
    assert descriptor.resize_support == "requires_chain_analysis"
    torch.testing.assert_close(gm(example), model(example))


def test_addmm_projection_can_grow_through_relu_to_functional_linear():
    from growingnn.actions.add_neurons import AddNeurons
    from growingnn.core.traced_model import TracedModel
    from growingnn.utils.fx import (
        functional_linear_resize_plan,
        resize_functional_linear_output,
    )

    class AddmmChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.producer_weight = nn.Parameter(torch.randn(4, 4))
            self.producer_bias = nn.Parameter(torch.randn(4))
            self.consumer_weight = nn.Parameter(torch.randn(3, 4))
            self.consumer_bias = nn.Parameter(torch.randn(3))

        def forward(self, value):
            hidden = torch.addmm(self.producer_bias, value, self.producer_weight)
            hidden = torch.relu(hidden)
            return torch.nn.functional.linear(
                hidden, self.consumer_weight, self.consumer_bias
            )

    model = AddmmChain().eval()
    gm = fx.symbolic_trace(model)
    value = torch.randn(2, 4)
    ShapeProp(gm).propagate(value)
    producer = next(
        node for node in gm.graph.nodes
        if node.op == "call_function" and "addmm" in str(node.target)
    )
    plan, reason = functional_linear_resize_plan(gm, producer, 6)

    assert reason is None
    assert plan is not None
    assert plan.source.weight_orientation == "in_features_by_out_features"
    assert plan.sink.form == "function"

    expected = gm(value)
    resize_functional_linear_output(gm, producer, 6)
    actual = gm(value)
    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.isfinite(actual).all()
    assert gm.producer_weight.shape == (4, 6)
    assert gm.producer_bias.shape == (6,)
    assert gm.consumer_weight.shape == (3, 6)
    assert all(
        "val" not in node.meta and "tensor_meta" not in node.meta
        for node in (producer, *plan.pointwise_nodes, plan.sink.node)
    )

    action_model = fx.symbolic_trace(model)
    traced = TracedModel.create(action_model, (2, 4), example_args=(value,))
    generated = AddNeurons.generate_all_actions(traced, ratio=1.5)
    action = next(
        action for action in generated
        if len(action.params) > 2 and action.params[2] == "fx_functional"
        and action.params[0] == producer.name
    )
    action.execute(traced)
    assert traced.gm.producer_weight.shape == (4, 6)
    assert tuple(traced.gm(value).shape) == (2, 3)


def test_transposed_addmm_producer_and_consumer_growth_use_stored_weight_orientation():
    from growingnn.utils.fx import (
        functional_linear_resize_plan,
        resize_functional_linear_output,
    )

    class TransposedAddmmChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.producer_weight = nn.Parameter(torch.randn(5, 4))
            self.producer_bias = nn.Parameter(torch.randn(5))
            self.consumer_weight = nn.Parameter(torch.randn(3, 5))
            self.consumer_bias = nn.Parameter(torch.randn(3))

        def forward(self, value):
            hidden = torch.addmm(
                self.producer_bias, value, self.producer_weight.t()
            )
            hidden = torch.relu(hidden)
            return torch.addmm(
                self.consumer_bias, hidden, self.consumer_weight.t()
            )

    model = TransposedAddmmChain().eval()
    gm = fx.symbolic_trace(model)
    value = torch.randn(2, 4)
    ShapeProp(gm).propagate(value)
    producers = [
        node for node in gm.graph.nodes
        if node.op == "call_function" and "addmm" in str(node.target)
    ]
    producer = producers[0]
    plan, reason = functional_linear_resize_plan(gm, producer, 7)

    assert reason is None
    assert plan is not None
    assert plan.source.weight_orientation == "out_features_by_in_features"
    assert plan.sink.form == "addmm"
    assert plan.sink.weight_orientation == "out_features_by_in_features"

    resize_functional_linear_output(gm, producer, 7)
    result = gm(value)
    assert tuple(result.shape) == (2, 3)
    assert torch.isfinite(result).all()
    assert gm.producer_weight.shape == (7, 4)
    assert gm.producer_bias.shape == (7,)
    assert gm.consumer_weight.shape == (3, 7)


def test_export_and_traced_model_support_keyword_only_example_inputs():
    from growingnn.core.traced_model import TracedModel

    class KeywordInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3)

        def forward(self, x, *, scale):
            return self.proj(x * scale)

    model = KeywordInputModel().eval()
    x = torch.randn(2, 4)
    scale = torch.tensor(0.75)
    traced = TracedModel.create(
        model,
        (2, 4),
        capture="export",
        example_args=(x,),
        example_kwargs={"scale": scale},
    )

    traced.shapes()
    output = next(node for node in traced.gm.graph.nodes if node.op == "output")

    output_metadata = output.meta["tensor_meta"]
    if isinstance(output_metadata, tuple):
        output_metadata = output_metadata[0]
    assert tuple(output_metadata.shape) == (2, 3)
    torch.testing.assert_close(traced.gm(x, scale), model(x, scale=scale))


def test_shape_propagation_invalidates_region_and_action_site_metadata_cache():
    from growingnn.core.traced_model import TracedModel

    class FunctionalTail(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return torch.relu(self.proj(x))

    traced = TracedModel.create(fx.symbolic_trace(FunctionalTail()), (2, 4))
    sites_without_metadata = traced.action_sites()
    assert sites_without_metadata
    assert not any(site.eligible for site in sites_without_metadata)

    traced.shapes()
    sites_with_metadata = traced.action_sites()
    assert any(site.eligible for site in sites_with_metadata)


def test_export_capture_accepts_keyword_only_inputs_without_positional_examples():
    from growingnn.utils.fx.capture import capture_graph

    class KeywordInputModel(nn.Module):
        def forward(self, *, x):
            return torch.relu(x)

    x = torch.randn(2, 4)
    gm = capture_graph(
        KeywordInputModel().eval(), route="export", example_kwargs={"x": x}
    )

    torch.testing.assert_close(gm(x), torch.relu(x))


def test_nested_region_discovery_tracks_torch_cond_branch_graphs():
    class ConditionalModel(nn.Module):
        def forward(self, value):
            predicate = value.sum() > 0
            return torch.cond(
                predicate,
                lambda operand: torch.sin(operand),
                lambda operand: torch.cos(operand),
                (value,),
            )

    from growingnn.utils.fx import capture_graph

    model = ConditionalModel().eval()
    example = torch.randn(2, 3)
    gm = capture_graph(model, route="export", example_args=(example,))
    torch.testing.assert_close(gm(example), model(example))

    from growingnn.core.traced_model import TracedModel

    traced = TracedModel.create(gm, (2, 3), example_args=(example,))
    nested = traced.nested_regions()
    branch_paths = {region.graph_path for region in nested if region.graph_path}
    assert any(path[-1].startswith("true_graph") for path in branch_paths)
    assert any(path[-1].startswith("false_graph") for path in branch_paths)
    assert any(
        region.graph_path
        and any(node.op == "call_function" and "sin" in str(node.target) for node in region.nodes)
        for region in nested
    )

    # Root action sites remain scoped to the root graph. Branch-aware edits
    # need an explicit subgraph rewriter rather than accidental cross-graph edits.
    root_sites = GraphRegionAnalyzer.discover_action_sites(gm)
    assert all(site.region.entry.graph is gm.graph for site in root_sites)


def test_export_regions_ignore_get_attr_roots_for_activation_dominance():
    from growingnn.utils.fx.capture import capture_graph

    model = _Residual().eval()
    example = torch.randn(2, 4)
    gm = capture_graph(model, route="export", example_args=(example,))
    regions = GraphRegionAnalyzer.discover(gm)

    # Export lowers Linear modules into functional ATen nodes with weight and
    # bias get_attr inputs. Those parameter edges remain explicit boundaries,
    # while activation dominance still discovers multi-operation regions.
    multi_operation_regions = [region for region in regions if len(region.nodes) > 1]
    assert multi_operation_regions
    assert any(
        any(node.op == "call_function" for node in region.nodes)
        for region in multi_operation_regions
    )
    assert any(
        edge.source is not None
        and edge.source.op == "get_attr"
        and edge.kind == "parameter_or_buffer"
        for region in multi_operation_regions
        for edge in region.inputs
    )
    assert any(
        region.structurally_valid
        and any(node.op == "call_function" for node in region.nodes)
        for region in multi_operation_regions
    )


def test_export_module_stack_exposes_nested_block_attention_and_mlp_regions():
    from growingnn.utils.fx.capture import capture_graph

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)
            self.k = nn.Linear(4, 4)
            self.v = nn.Linear(4, 4)
            self.out = nn.Linear(4, 4)

        def forward(self, x):
            return self.out(self.q(x) + self.k(x) + self.v(x))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 8)
            self.down = nn.Linear(8, 4)

        def forward(self, x):
            return self.down(torch.nn.functional.gelu(self.up(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.mlp = FeedForward()

        def forward(self, x):
            x = x + self.attn(x)
            return x + self.mlp(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block()])

        def forward(self, x):
            return self.blocks[0](x)

    gm = capture_graph(Model().eval(), route="export", example_args=(torch.randn(2, 4),))
    regions = GraphRegionAnalyzer.discover(gm)
    groups = {}
    for node in gm.graph.nodes:
        stack = node.meta.get("nn_module_stack", {})
        if not isinstance(stack, dict):
            continue
        for entry in stack.values():
            if isinstance(entry, (tuple, list)) and entry and entry[0] in {
                "blocks.0", "blocks.0.attn", "blocks.0.mlp"
            } and isinstance(node.meta.get("val"), torch.Tensor):
                groups.setdefault(entry[0], set()).add(node)

    for path in ("blocks.0", "blocks.0.attn", "blocks.0.mlp"):
        assert path in groups and len(groups[path]) > 1
        assert any(groups[path].issubset(set(region.nodes)) for region in regions)


def test_region_discovery_keeps_nested_module_groups_with_shared_endpoints():
    graph = fx.Graph()
    root = nn.Module()
    x = graph.placeholder("x")
    first = graph.call_function(torch.neg, args=(x,))
    middle = graph.call_function(torch.relu, args=(first,))
    last = graph.call_function(torch.sigmoid, args=(middle,))
    graph.output(last)
    gm = fx.GraphModule(root, graph)
    stack = {
        "model": ("model", "Model"),
        "block": ("model.block", "Block"),
        "inner": ("model.block.inner", "Inner"),
    }
    for node in (first, middle, last):
        node.meta["val"] = torch.randn(2, 3)
    first.meta["nn_module_stack"] = stack
    middle.meta["nn_module_stack"] = {"model": stack["model"], "block": stack["block"]}
    last.meta["nn_module_stack"] = stack

    regions = GraphRegionAnalyzer.discover(gm)
    block = next(region for region in regions if "model.block" in region.module_paths)
    inner = next(region for region in regions if "model.block.inner" in region.module_paths)

    assert block.entry is first and block.exit is last
    assert inner.entry is first and inner.exit is last
    assert set(inner.nodes) == {first, last}
    assert set(block.nodes) == {first, middle, last}


def test_region_discovery_handles_deep_functional_graph_without_python_recursion():
    graph = fx.Graph()
    value = graph.placeholder("x")
    for _ in range(10000):
        value = graph.call_function(torch.neg, args=(value,))
    graph.output(value)
    gm = fx.GraphModule(nn.Module(), graph)

    candidates = GraphRegionAnalyzer.discover(gm)

    assert len(candidates) == 10000
    assert all(len(candidate.nodes) == 1 for candidate in candidates)


def test_low_rank_feature_adapter_preserves_interface_and_scales_to_wide_features():
    from growingnn.utils.fx.adapters import LowRankFeatureAdapter, supported_adapter_ranks

    value = torch.randn(2, 5, 64)
    edge_adapter = LowRankFeatureAdapter(64, 8, residual=True)
    branch_adapter = LowRankFeatureAdapter(64, 8, residual=False)
    torch.testing.assert_close(edge_adapter(value), value)
    torch.testing.assert_close(branch_adapter(value), torch.zeros_like(value))
    assert edge_adapter(value).shape == value.shape
    channels_first = torch.randn(2, 64, 8, 8)
    channel_adapter = LowRankFeatureAdapter(64, 8, residual=True, feature_axis=1)
    torch.testing.assert_close(channel_adapter(channels_first), channels_first)
    assert supported_adapter_ranks(4096, 1_000_000) == (4, 8, 16)


def test_feature_interface_separates_axis_policy_from_tensor_metadata():
    from growingnn.utils.fx.regions import ValueInterface, analyze_feature_tensor

    sequence = analyze_feature_tensor(
        ValueInterface(
            kind="tensor", shape=(2, 12, 64), dtype="torch.float32", device="cpu"
        )
    )
    assert sequence.eligible
    assert sequence.feature_axis == 2
    assert sequence.features == 64
    assert sequence.dtype is torch.float32

    image = analyze_feature_tensor(
        ValueInterface(
            kind="tensor", shape=(2, 64, 16, 16), dtype="torch.float32", device="cpu"
        ),
        feature_axis=1,
        supported_ranks=(4,),
    )
    assert image.eligible
    assert image.feature_axis == 1
    assert image.features == 64

    mixed_policy = {2: -1, 3: -1, 4: 1}
    mixed = analyze_feature_tensor(
        ValueInterface(
            kind="tensor", shape=(2, 64, 8, 8), dtype="torch.float32", device="cpu"
        ),
        feature_axis=mixed_policy,
    )
    assert mixed.eligible
    assert mixed.feature_axis == 1
    assert mixed.features == 64

    unsupported = analyze_feature_tensor(
        ValueInterface(kind="tensor", shape=(2, 64, 16, 16), dtype="torch.float32")
    )
    assert not unsupported.eligible
    assert "no configured feature-axis policy" in unsupported.rejection_reason


def test_region_enumeration_stays_linear_on_long_module_chain():
    graph = fx.Graph()
    root = nn.Module()
    value = graph.placeholder("x")
    for index in range(300):
        name = f"layer_{index}"
        root.add_module(name, nn.Identity())
        value = graph.call_module(name, args=(value,))
    graph.output(value)
    gm = fx.GraphModule(root, graph)

    regions = GraphRegionAnalyzer.discover(gm)

    # Single-operation regions plus one adjacent nested candidate per layer edge.
    assert len(regions) <= 2 * 300
    assert any(len(region.nodes) == 2 for region in regions)
