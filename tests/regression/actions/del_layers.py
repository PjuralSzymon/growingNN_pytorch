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

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.delete_layer import DelLayer, explain_delete_layer_blockers
from growingnn.utils.fx import GraphStructureQuery, GraphConnectivity
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
    model = ModelFactory.complex_residual_many_widths()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    output_initial = gm(x)
    norms = []
    parameter_amounts = []
    parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))

    # Act
    iterantions = 50
    grow_iterations = int(iterantions/2)-1
    #TODO handling those 0 and +1 should be also added to other regresison tests
    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph0", fmt="pdf")
    for id in range(iterantions):
        logger.info("idx: %s --------------------------------", id)
        actions: List[Action] = []
        if id < grow_iterations:
            actions += AddResLinearLayer.generate_all_actions(gm, layer_types=[Layer_Type.EYE])
            actions += AddResConvLayer.generate_all_actions(gm)
            actions += AddSeqConvLayer.generate_all_actions(gm)
            actions += AddSeqLinearLayer.generate_all_actions(gm)
        else:
            actions += DelLayer.generate_all_actions(gm)
        if len(actions) == 0:
            logger.warning("No actions to execute for iteration %s", id)
            break
        idx = rng.randrange(len(actions))
        logger.info("action used: %s", actions[idx])
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
        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id+1), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id+1), fmt="pdf")
        logger.info("diffrence norm: %s", dn)

    all_modules = sorted({str(n.target) for n in gm.graph.nodes if n.op == "call_module"})
    hidden_modules = list(dict.fromkeys(GraphStructureQuery.get_all_hidden_modules(gm)))
    deletable = DelLayer.generate_all_actions(gm)
    blockers = explain_delete_layer_blockers(gm)
    logger.info("final parameter count: %s", GraphStructureQuery.get_amount_of_parameters(gm))
    logger.info("final call_module layers (%s): %s", len(all_modules), all_modules)
    logger.info("final hidden layers (%s): %s", len(hidden_modules), hidden_modules)
    logger.info("final deletable layers (%s): %s", len(deletable), [a.params[0] for a in deletable])
    for layer_id, reason in blockers:
        logger.info("blocked delete %s: %s", layer_id, reason)
    for issue in GraphConnectivity.explain_connectivity(gm):
        logger.info("graph connectivity: %s", issue)

    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
