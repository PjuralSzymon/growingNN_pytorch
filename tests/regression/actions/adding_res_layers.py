import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.utils.fx import GraphStructureQuery, extract_graph
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
if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelFactory.complex_residual_many_widths()
    gm = extract_graph(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    output_initial = gm(x)
    norms = []
    parameter_amounts = []
    parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))

    # Act
    id = 0
    for _ in range(50):
        actions: List[AddResLinearLayer] = AddResLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)), layer_types=[Layer_Type.EYE])
        id += 1
        idx = rng.randrange(len(actions))
        logger.info("idx: %s --------------------------------", id)
        logger.info("action used: %s", actions[idx])
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id), fmt="pdf")
        actions[idx].execute(TracedModel.create(gm, (1, 4)))
        output_final = gm(x)
        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
        logger.info("diffrence norm: %s", dn)

    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
