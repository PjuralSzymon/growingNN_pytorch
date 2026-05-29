"""General-purpose torch.fx utilities for graph inspection and mutation."""

from growingnn.utils.fx.node_analysis import ModuleResolver, NodeTypeChecker, NodeWidthAnalyser
from growingnn.utils.fx.node_editor import NodeEditor
from growingnn.utils.fx.graph_analysis import (
    ModuleClassifier,
    GraphStructureQuery,
    LayerShapeAnalyser,
    LayerBridgeFinder,
)
from growingnn.utils.fx.graph_editor import ModelStructureEditor
