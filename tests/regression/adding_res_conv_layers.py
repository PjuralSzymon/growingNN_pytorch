import sys
from pathlib import Path

# Repo root must be on sys.path before any `growingnn` / `tests` imports (script may be run from any cwd).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import random
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.model_factory import ModelFactory
from tests.regression.regression_utils import FOLDER_NAME, clear_regression_folder, parse_regression_cli


if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4, 8, 8)
    rng = random.Random(42)
    y = gm(x)
    norms = []

    # Act
    id = 0
    for _ in range(50):
        actions: List[AddResConvLayer] = AddResConvLayer.generate_all_actions(gm)
        id += 1
        idx = rng.randrange(len(actions))
        print(f"idx: {id} " + "--------------------------------")
        print(f"gm.graph: {gm.graph}")
        print("action used: ", actions[idx])
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id), fmt="pdf")
        actions[idx].execute(gm)


        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified_Q" + str(id), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_Q" + str(id), fmt="pdf")

        y_new = gm(x)
        dn = float(torch.norm(y - y_new))
        norms.append(dn)
        print(f"diffrence norm: {dn}")

    plt.plot(range(len(norms)), norms)
    plt.ylabel("||Δout||")
    plt.xlabel("step")
    plt.show()

    if not args.save_output:
        clear_regression_folder()
