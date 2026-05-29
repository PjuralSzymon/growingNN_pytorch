import sys
from pathlib import Path


# Repo root must be on sys.path before any `growingnn` / `tests` imports (script may be run from any cwd).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import random
from typing import List

import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.utils.fx import GraphStructureQuery

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.core.logger import logger
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.model_factory import ModelFactory
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)


if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelFactory.complex_residual_conv_many_widths()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4, 8, 8)
    rng = random.Random(42)
    y = gm(x)
    norms = []
    parameter_amounts = []
    parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))

    # Act
    for id in range(30):
        logger.info("idx: %s --------------------------------", id)
        logger.debug("gm.graph: %s", gm.graph)
        actions: List[AddSeqConvLayer] = AddSeqConvLayer.generate_all_actions(gm)
        idx = rng.randrange(len(actions))
        logger.info("action used: %s", actions[idx])
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id), fmt="pdf")
        actions[idx].execute(gm)

        try:
            output_final = gm(x)
        except Exception:
            draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified_error" + str(id+1), fmt="pdf")
            logger.exception(
                "Error executing action %s",
                actions[idx],
            )
            break

        dn = float(torch.norm(y - output_final))
        norms.append(dn)
        parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
        logger.info("diffrence norm: %s", dn)

    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
