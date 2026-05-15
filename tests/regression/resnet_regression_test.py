from doctest import FAIL_FAST
import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.fx as fx
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.maxvit import F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_layer import AddSeqLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.utils.model_analyser import get_amount_of_parameters
from growingnn.core.logger import logger
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)


# Which growth actions to consider (delete is always available in the shrink phase).
USE_ADD_RES_LAYER = True
USE_ADD_RES_CONV_LAYER = True
USE_ADD_SEQ_LAYER = False
USE_ADD_SEQ_CONV_LAYER = False
USE_DEL_LAYER = False

BATCH_SIZE = 2
INPUT_SHAPE = (3, 64, 64)
ITERATIONS = 50


def _load_pretrained_resnet18() -> torch.nn.Module:
    """Real pretrained ResNet-18 from torchvision. ``eval()`` to keep BN running stats fixed."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model


def _make_xy(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Small CHW batch in the right shape; labels in ``[0, 1000)`` (ImageNet classes)."""
    x = torch.randn(BATCH_SIZE, *INPUT_SHAPE, generator=rng)
    y = torch.randint(0, 1000, (BATCH_SIZE,), generator=rng)
    return x, y


def _generate_actions(gm: fx.GraphModule) -> List[Action]:
    actions: List[Action] = []
    if USE_ADD_RES_LAYER:
        actions += AddResLayer.generate_all_actions(gm, layer_types=[Layer_Type.EYE])
    if USE_ADD_RES_CONV_LAYER:
        actions += AddResConvLayer.generate_all_actions(gm)
    if USE_ADD_SEQ_LAYER:
        actions += AddSeqLayer.generate_all_actions(gm)
    if USE_ADD_SEQ_CONV_LAYER:
        actions += AddSeqConvLayer.generate_all_actions(gm)
    if USE_DEL_LAYER:
        actions += DelLayer.generate_all_actions(gm)
    return actions

if __name__ == "__main__":
    args = parse_regression_cli()

    model = _load_pretrained_resnet18()
    gm = fx.symbolic_trace(model)

    rng = random.Random(42)
    data_rng = torch.Generator().manual_seed(42)
    x, _ = _make_xy(data_rng)

    with torch.no_grad():
        output_initial = gm(x)

    norms: List[float] = []
    parameter_amounts: List[int] = [get_amount_of_parameters(gm)]


    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph0", fmt="pdf")
    logger.info("Initial ResNet-18 Graph loaded and saved")

    for id in range(ITERATIONS):
        logger.info("idx: %s --------------------------------", id)
        actions = _generate_actions(gm)

        if len(actions) == 0:
            logger.warning("No actions to execute for iteration %s", id)
            break

        idx = rng.randrange(len(actions))
        logger.info("action used: %s", actions[idx])
        try:
            actions[idx].execute(gm)
            with torch.no_grad():
                output_final = gm(x)
        except Exception:
            draw_filtered_fx_graph(
                gm, FOLDER_NAME + "/" + "fx_graph_simplified_error" + str(id + 1), fmt="pdf"
            )
            logger.exception("Error executing action %s", actions[idx])
            break

        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        parameter_amounts.append(get_amount_of_parameters(gm))
        draw_filtered_fx_graph(
            gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id + 1), fmt="pdf"
        )
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id + 1), fmt="pdf")
        logger.info("diffrence norm: %s", dn)

    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
