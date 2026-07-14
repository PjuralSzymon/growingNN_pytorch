import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Action
from growingnn.actions.add_neurons import AddNeurons
from growingnn.utils.fx import GraphStructureQuery
from growingnn.core.logger import logger
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)
_CONV_TRACE_SHAPE = (1, 4, 8, 8)


if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelFactory.complex_residual_conv_many_widths()
    gm = fx.symbolic_trace(model)
    x = torch.randn(2, 4, 8, 8)
    rng = random.Random(42)
    output_initial = gm(x)
    norms = []
    parameter_amounts = [GraphStructureQuery.get_amount_of_parameters(gm)]

    iterations = 3
    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph0", fmt="pdf")

    for id in range(iterations):
        logger.info("idx: %s --------------------------------", id)
        actions: List[Action] = AddNeurons.generate_all_actions(TracedModel.create(gm, _CONV_TRACE_SHAPE))
        if len(actions) == 0:
            logger.warning("No actions to execute for iteration %s", id)
            break
        idx = rng.randrange(len(actions))
        logger.info("action used: %s", actions[idx])
        actions[idx].execute(TracedModel.create(gm, _CONV_TRACE_SHAPE))
        try:
            output_final = gm(x)
        except Exception:
            draw_filtered_fx_graph(
                gm, FOLDER_NAME + "/" + "fx_graph_simplified_error" + str(id + 1), fmt="pdf"
            )
            logger.info("gm.graph: %s", gm.graph)
            logger.info("actions: %s", actions)
            logger.info("idx: %s", idx)
            logger.info("actions[idx]: %s", actions[idx])
            logger.info("norms: %s", norms)
            logger.info("parameter_amounts: %s", parameter_amounts)
            logger.exception("Error executing action %s", actions[idx])
            break
        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id + 1), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id + 1), fmt="pdf")
        logger.info("diffrence norm: %s", dn)

    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
