import random
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.model_factory import ModelFactory
from tests.regression.regression_utils import FOLDER_NAME, clear_regression_folder, parse_regression_cli


if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    output_initial = gm(x)
    norms = []

    # Act
    id = 0
    for _ in range(50):
        actions: List[AddResLayer] = AddResLayer.generate_all_actions(gm, layer_types=[Layer_Type.EYE])
        id += 1
        idx = rng.randrange(len(actions))
        print(f"idx: {id} " + "--------------------------------")
        print(f"gm.graph: {gm.graph}")
        print("action used: ", actions[idx])
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id), fmt="pdf")
        actions[idx].execute(gm)
        output_final = gm(x)
        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        print(f"diffrence norm: {dn}")

    plt.plot(range(len(norms)), norms)
    plt.ylabel("||Δout||")
    plt.xlabel("step")
    plt.show()

    if not args.save_output:
        clear_regression_folder()
