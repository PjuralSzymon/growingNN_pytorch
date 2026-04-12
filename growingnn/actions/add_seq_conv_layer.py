
from typing import List

from torch import fx, nn
from .action import Action, Layer_Type

from growingnn.actions.utils.layer_Factory import ConvFactory, LinearFactory
from growingnn.actions.utils.model_analyser import module_dependency_pairs, module_sequential_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer, add_new_seq_layer
from .action import Action, Layer_Type


class AddSeqConvLayer(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_seq_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        actions : List[Action] = []
        name_prefix = "seq_linear"
        layer_types = (nn.modules.conv._ConvNd, nn.modules.AdaptiveAvgPool2d, nn.modules.AdaptiveMaxPool2d, nn.modules.AdaptiveAvgPool1d, nn.modules.AdaptiveMaxPool1d)
        pairs = module_sequential_pairs(model)
        for layer_from_id, layer_to_id in pairs:
            layer_from = getattr(model, layer_from_id, None)
            layer_to = getattr(model, layer_to_id, None)

            if  isinstance(layer_from, nn.modules.conv._ConvNd):
                print(layer_to.type)
                if  isinstance(layer_to, layer_types):
                    name = unique_call_module_name(name_prefix, model)
                    layer = ConvFactory.create_zero_conv(
                        in_channels=layer_from.out_channels,
                        out_channels=layer_from.out_channels,
                        kernel_size=layer_from.kernel_size,
                        stride=1,
                        padding=layer_from.padding
                    )
                    actions.append(AddSeqConvLayer([layer_from_id, layer_to_id, layer, name]))
        return actions
    
    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "