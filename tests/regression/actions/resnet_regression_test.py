from doctest import FAIL_FAST
import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.fx as fx
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.maxvit import F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer
from growingnn.actions.add_neurons import AddNeurons
from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.utils.fx import GraphStructureQuery
from growingnn.core.logger import logger
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from growingnn.actions.delete_neurons import DelNeurons
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    log_regression_action_error,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)


# Which growth actions to consider (delete is always available in the shrink phase).
USE_ADD_RES_LAYER = False
USE_ADD_RES_CONV_LAYER = False
USE_ADD_SEQ_LAYER = False
USE_ADD_SEQ_CONV_LAYER = False
USE_DEL_LAYER = False
USE_DEL_NEURONS = False
USE_ADD_NEURONS = False
USE_ADD_SEQ_DROPOUT = True

BATCH_SIZE = 2
INPUT_SHAPE = (3, 64, 64)
ITERATIONS = 20


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
        actions += AddResLinearLayer.generate_all_actions(gm, layer_types=[Layer_Type.EYE])
    if USE_ADD_RES_CONV_LAYER:
        actions += AddResConvLayer.generate_all_actions(gm)
    if USE_ADD_SEQ_LAYER:
        actions += AddSeqLinearLayer.generate_all_actions(gm)
    if USE_ADD_SEQ_CONV_LAYER:
        actions += AddSeqConvLayer.generate_all_actions(gm)
    if USE_DEL_LAYER:
        actions += DelLayer.generate_all_actions(gm)
    if USE_DEL_NEURONS:
        actions += DelNeurons.generate_all_actions(gm)
    if USE_ADD_NEURONS:
        actions += AddNeurons.generate_all_actions(gm)
    if USE_ADD_SEQ_DROPOUT:
        actions += AddSeqDropoutLayer.generate_all_actions(model, p=0.1)
    return actions

def _generate_only_shrink_actions(gm: fx.GraphModule) -> List[Action]:
    actions: List[Action] = []
    if USE_DEL_NEURONS:
        actions += DelNeurons.generate_all_actions(gm)
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
    parameter_amounts: List[int] = [GraphStructureQuery.get_amount_of_parameters(gm)]
    used_action_types: List[str] = []

    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph0", fmt="pdf")
    logger.info("Initial ResNet-18 Graph loaded and saved")

    id = 0
    while True:
        logger.info("idx: %s --------------------------------", id)
        if id >= ITERATIONS:
            actions = _generate_only_shrink_actions(gm)
        else:
            actions = _generate_actions(gm)
        id += 1
        
        if len(actions) == 0:
            logger.warning("No actions to execute for iteration %s", id)
            break

        idx = rng.randrange(len(actions))
        chosen = actions[idx]
        used_action_types.append(type(chosen).__name__)
        logger.info("action type: %s", type(chosen).__name__)
        logger.info("action used idx: %s [%s]: %s", idx, used_action_types[-1], chosen)
        try:
            chosen.execute(gm)
            with torch.no_grad():
                output_final = gm(x)
        except Exception:
            draw_filtered_fx_graph(
                gm, FOLDER_NAME + "/" + "fx_graph_simplified_error" + str(id + 1), fmt="pdf"
            )
            log_regression_action_error(
                gm,
                chosen,
                actions=actions,
                idx=idx,
                norms=norms,
                parameter_amounts=parameter_amounts,
            )
            break

        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
        draw_filtered_fx_graph(
            gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id + 1), fmt="pdf"
        )
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id + 1), fmt="pdf")
        logger.info("diffrence norm: %s", dn)

    action_counts: dict[str, int] = {}
    for name in used_action_types:
        action_counts[name] = action_counts.get(name, 0) + 1
    logger.info("action summary (%d total):", len(used_action_types))
    col = max((len(name) for name in action_counts), default=6)
    logger.info("%-*s | %s", col, "action", "count")
    logger.info("%s-+-%s", "-" * col, "-" * 5)
    for name in sorted(action_counts):
        logger.info("%-*s | %d", col, name, action_counts[name])

    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
