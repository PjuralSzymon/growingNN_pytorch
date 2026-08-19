"""
GitHub #13 regression: grow a nested residual graph, then shrink neurons.

Issue claim: simple graphs often look fine; as the graph grows and becomes nested,
one layer change forces a huge cascade and propagation becomes unstable.

Phase A: AddResLinearLayer + AddSeqLinearLayer (build nesting).
Phase B: DelNeurons only (pressure on fix_graph_widths).

Run:
  python tests/regression/actions/layer_resize_nested_grow_shrink.py
  python tests/regression/actions/layer_resize_nested_grow_shrink.py --save-output true

What to look for:
  - forward crash after a DelNeurons cascade
  - param count / width logs jumping wildly
  - HEAD BROKEN if the 4-d head is touched
  - error PDF graphs at the failing step
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.logger import logger
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import GraphStructureQuery
from tests.model_factory import ModelFactory
from tests.regression.actions.layer_resize_regression_common import (
    check_head_keeps_class_count,
    draw_step_graphs,
    fail_action_context,
    find_head_linear_name,
    log_linear_widths,
)
from tests.regression.regression_utils import (
    clear_regression_folder,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)

TRACE_SHAPE = (1, 4)
GROW_ITERATIONS = 20
SHRINK_ITERATIONS = 40
SEED = 42
# Head of complex_residual_many_widths is Linear(11, 4) — treat 4 as "class count".
N_CLASSES = 4
SHRINK_RATIO = 0.5


if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelFactory.complex_residual_many_widths()
    gm = fx.symbolic_trace(model)
    x = torch.randn(2, 4)
    rng = random.Random(SEED)
    head_name = find_head_linear_name(gm)
    output_initial = gm(x)
    norms: list[float] = []
    parameter_amounts = [GraphStructureQuery.get_amount_of_parameters(gm)]
    ok = True

    logger.info(
        "=== nested grow then shrink === head=%s grow=%s shrink=%s",
        head_name,
        GROW_ITERATIONS,
        SHRINK_ITERATIONS,
    )
    log_linear_widths(gm, tag="start")
    draw_step_graphs(gm, 0)

    total = GROW_ITERATIONS + SHRINK_ITERATIONS
    for step in range(1, total + 1):
        logger.info("idx: %s --------------------------------", step)
        growing = step <= GROW_ITERATIONS
        actions: List[Action] = []
        if growing:
            traced = TracedModel.create(gm, TRACE_SHAPE)
            actions += AddResLinearLayer.generate_all_actions(traced, layer_types=[Layer_Type.EYE])
            actions += AddSeqLinearLayer.generate_all_actions(traced)
            phase = "grow"
        else:
            actions += DelNeurons.generate_all_actions(
                TracedModel.create(gm, TRACE_SHAPE),
                ratio=SHRINK_RATIO,
            )
            phase = "shrink"

        if not actions:
            logger.warning("No %s actions at step %s", phase, step)
            if growing:
                continue
            break

        idx = rng.randrange(len(actions))
        chosen = actions[idx]
        logger.info("phase=%s action used: %s", phase, chosen)
        chosen.execute(TracedModel.create(gm, TRACE_SHAPE))
        log_linear_widths(gm, tag=f"after_{phase}_{step}")

        try:
            output_final = gm(x)
        except Exception:
            ok = False
            fail_action_context(
                gm,
                chosen,
                actions=actions,
                idx=idx,
                norms=norms,
                parameter_amounts=parameter_amounts,
                step=step,
                phase=phase,
            )
            break

        if not check_head_keeps_class_count(gm, N_CLASSES, head_name=head_name, step=step):
            ok = False
            draw_step_graphs(gm, step, error=True)
            break

        dn = float(torch.norm(output_initial - output_final))
        norms.append(dn)
        parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
        draw_step_graphs(gm, step)
        logger.info("diffrence norm: %s params=%s", dn, parameter_amounts[-1])

    plot_norms_and_parameter_count(norms, parameter_amounts)
    if ok:
        logger.info(
            "layer_resize_nested_grow_shrink: finished ok start_params=%s end_params=%s",
            parameter_amounts[0],
            parameter_amounts[-1],
        )
    else:
        logger.error("layer_resize_nested_grow_shrink: FAILED")

    if not args.save_output:
        clear_regression_folder()
