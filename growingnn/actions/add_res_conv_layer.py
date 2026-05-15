from typing import List

from torch import ceil, clip, floor, fx, nn
import torch

from growingnn.actions.utils.conv_to_linear_adapter import can_insert_conv_before_linear
from growingnn.actions.utils.fx_shape_probe import call_module_output_shapes
from growingnn.actions.utils.layer_Factory import ConvFactory, LinearFactory
from growingnn.actions.utils.model_analyser import get_layer_module, module_dependency_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer
from .action import Action, Layer_Type


class AddResConvLayer(Action):

    SUPPORTED_MODULES_FROM_LAYER = (nn.modules.conv._ConvNd,)
    SUPPORTED_MODULES_TO_LAYER = (nn.modules.conv._ConvNd, nn.modules.Linear)

    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])
    
    def can_be_infulenced(self, by_action):
        return False

    def get_conv_output_shape(layer: nn.modules.conv._ConvNd, spatial_size):
        in_channels = layer.in_channels
        x = torch.randn(1, in_channels, *spatial_size)
        with torch.no_grad():
            y = layer(x)
        return tuple(y.shape)

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        actions : List[Action] = []
        name_prefix = "res_conv_"
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        pairs = module_dependency_pairs(gm)
        out_shapes = call_module_output_shapes(gm)
        for layer_from_id, layer_to_id in pairs:
            layer_from = get_layer_module(layer_from_id, model)
            layer_to = get_layer_module(layer_to_id, model)
            if not isinstance(layer_from, AddResConvLayer.SUPPORTED_MODULES_FROM_LAYER):
                continue
            if not isinstance(layer_to, AddResConvLayer.SUPPORTED_MODULES_TO_LAYER):
                continue

            name = unique_call_module_name(name_prefix, gm)
            if isinstance(layer_to, nn.modules.conv._ConvNd):
                if out_shapes:
                    s_from = out_shapes.get(layer_from_id)
                    s_to = out_shapes.get(layer_to_id)
                    if s_from is None or s_to is None or s_from != s_to:
                        continue
                # Residual is ``merge + proj(src)`` on *outputs*; proj out channels must match ``layer_to``.
                layer = ConvFactory.create_zero_conv(
                    in_channels=layer_from.out_channels,
                    out_channels=layer_to.out_channels,
                    kernel_size=layer_from.kernel_size,
                    stride=1,
                    padding=layer_from.padding,
                )
                actions.append(AddResConvLayer([layer_from_id, layer_to_id, layer, name]))
            elif isinstance(layer_to, nn.modules.Linear):
                if can_insert_conv_before_linear(
                    layer_from.out_channels, layer_to.in_features
                ):
                    # Sum is on ``layer_to`` *output* (``out_features``); flattened conv must match that width.
                    layer = ConvFactory.create_zero_conv_before_linear(
                        in_channels=layer_from.out_channels,
                        out_channels=layer_to.out_features,
                        kernel_size=layer_from.kernel_size,
                        stride=1,
                        padding=layer_from.padding,
                    )
                    actions.append(AddResConvLayer([layer_from_id, layer_to_id, layer, name]))                                    
        return actions
    
    def __str__(self):
        return " ( Add Res Conv Layer Action: " + str(self.params) + " ) "