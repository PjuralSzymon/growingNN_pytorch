from typing import List

from torch import ceil, clip, floor, fx, nn
import torch

from growingnn import config
from growingnn.actions.utils.conv_to_linear_adapter import can_insert_conv_before_linear
from growingnn.actions.utils.layer_Factory import ConvFactory, LinearFactory
from growingnn.actions.utils.model_analyser import module_dependency_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer
from .action import Action, Layer_Type


class AddResConvLayer(Action):

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
        pairs = module_dependency_pairs(model)
        for layer_from_id, layer_to_id in pairs:
            layer_from = getattr(model, layer_from_id, None)
            layer_to = getattr(model, layer_to_id, None)
            if  isinstance(layer_from, nn.modules.conv._ConvNd):
                name = unique_call_module_name(name_prefix, model)
                if  isinstance(layer_to, nn.modules.conv._ConvNd):
                    layer = ConvFactory.create_zero_conv(
                        in_channels=layer_from.out_channels,
                        out_channels=layer_from.out_channels,
                        kernel_size=layer_from.kernel_size,
                        stride=1,
                        padding=layer_from.padding
                    )
                    actions.append(AddResConvLayer([layer_from_id, layer_to_id, layer, name]))
                elif  isinstance(layer_to, nn.modules.Linear):
                    if can_insert_conv_before_linear(layer_from.out_channels, layer_to.in_features):
                        layer = ConvFactory.create_zero_conv_before_linear(
                            in_channels=layer_from.out_channels,
                            out_channels=layer_from.out_channels,
                            kernel_size=layer_from.kernel_size,
                            stride=1,
                            padding=layer_from.padding
                        )
                        actions.append(AddResConvLayer([layer_from_id, layer_to_id, layer, name]))
                                    
        return actions
    
    def __str__(self):
        return " ( Add Res Conv Layer Action: " + str(self.params) + " ) "