"""General-purpose torch.fx utilities for graph inspection and mutation."""

from growingnn.utils.fx.node_analysis import ModuleResolver, NodeTypeChecker, NodeWidthAnalyser
from growingnn.utils.fx.node_editor import NodeEditor
from growingnn.utils.fx.graph_analysis import (
    ModuleClassifier,
    GraphStructureQuery,
    GraphConnectivity,
    LayerShapeAnalyser,
    LayerBridgeFinder,
)
from growingnn.utils.fx.graph_editor import ModelStructureEditor
from growingnn.utils.fx.capture import ConfigurableTracer, capture_graph, order_example_inputs
from growingnn.utils.fx.regions import (
    BoundaryEdge,
    GraphRegionAnalyzer,
    RegionActionSite,
    RegionCandidate,
    ValueInterface,
    FeatureTensorInterface,
    analyze_feature_tensor,
    PairTraversalDecision,
    pair_traversal_decision,
)
from growingnn.utils.fx.adapters import (
    FeatureAxisConv2dAdapter,
    LowRankFeatureAdapter,
    supported_adapter_ranks,
)
from growingnn.utils.fx.operation_descriptors import (
    LinearOperationDescriptor,
    describe_linear_operations,
    is_functional_convolution_operation,
)
from growingnn.utils.fx.functional_linear_resize import (
    FunctionalLinearResizeAnalyzer,
    FunctionalLinearResizePlan,
    functional_linear_resize_plan,
    resize_functional_linear_output,
)
