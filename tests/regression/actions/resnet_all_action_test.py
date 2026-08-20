import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.fx as fx
from torchvision.models import ResNet18_Weights, resnet18

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.logger import logger
from growingnn.utils.fx import GraphStructureQuery, extract_graph
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from growingnn.core.traced_model import TracedModel
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    log_regression_action_error,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)
BATCH_SIZE = 100
INPUT_SHAPE = (3, 64, 64)
ITERATIONS = 20
SEED = 42

ActionGenerator = Tuple[str, Callable[[fx.GraphModule], List[Action]]]

_TRACE_SHAPE = (1, *INPUT_SHAPE)

ACTION_GENERATORS: List[ActionGenerator] = [
    (
        "AddResLinearLayer",
        lambda gm: AddResLinearLayer.generate_all_actions(
            TracedModel.create(gm, _TRACE_SHAPE), layer_types=[Layer_Type.EYE]
        ),
    ),
    ("AddResConvLayer", lambda gm: AddResConvLayer.generate_all_actions(TracedModel.create(gm, _TRACE_SHAPE))),
    ("AddSeqLinearLayer", lambda gm: AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, _TRACE_SHAPE))),
    ("AddSeqConvLayer", lambda gm: AddSeqConvLayer.generate_all_actions(TracedModel.create(gm, _TRACE_SHAPE))),
    ("DelLayer", lambda gm: DelLayer.generate_all_actions(TracedModel.create(gm, _TRACE_SHAPE))),
    ("DelNeurons", lambda gm: DelNeurons.generate_all_actions(TracedModel.create(gm, _TRACE_SHAPE))),
]


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


def _log_action_summary(action_counts: dict[str, int]) -> None:
    total = sum(action_counts.values())
    logger.info("action summary (%d total):", total)
    col = max((len(name) for name in action_counts), default=6)
    logger.info("%-*s | %s", col, "action", "count")
    logger.info("%s-+-%s", "-" * col, "-" * 5)
    for name in sorted(action_counts):
        logger.info("%-*s | %d", col, name, action_counts[name])


if __name__ == "__main__":
    args = parse_regression_cli()

    model = _load_pretrained_resnet18()
    gm = extract_graph(model)

    rng = random.Random(SEED)
    data_rng = torch.Generator().manual_seed(SEED)
    x, _ = _make_xy(data_rng)

    with torch.no_grad():
        output_initial = gm(x)

    norms: List[float] = []
    parameter_amounts: List[int] = [GraphStructureQuery.get_amount_of_parameters(gm)]
    action_counts: dict[str, int] = {name: 0 for name, _ in ACTION_GENERATORS}

    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified0", fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph0", fmt="pdf")
    logger.info("Initial ResNet-18 graph loaded and saved")

    step = 0
    for iteration in range(ITERATIONS):
        logger.info("iteration: %s --------------------------------", iteration)
        order = list(ACTION_GENERATORS)
        rng.shuffle(order)

        for action_name, generate in order:
            actions = generate(gm)
            if not actions:
                continue

            idx = rng.randrange(len(actions))
            chosen = actions[idx]
            logger.info(
                "iteration %s | %s | picked %s/%s: %s",
                iteration,
                action_name,
                idx,
                len(actions),
                chosen,
            )
            try:
                chosen.execute(TracedModel.create(gm, _TRACE_SHAPE))
                with torch.no_grad():
                    output_final = gm(x)
            except Exception:
                draw_filtered_fx_graph(
                    gm,
                    FOLDER_NAME + "/" + f"fx_graph_simplified_error_iter{iteration}_{action_name}",
                    fmt="pdf",
                )
                log_regression_action_error(
                    gm,
                    chosen,
                    actions=actions,
                    action_type=action_name,
                    norms=norms,
                    parameter_amounts=parameter_amounts,
                    action_counts=action_counts,
                )
                _log_action_summary(action_counts)
                plot_norms_and_parameter_count(norms, parameter_amounts)
                if not args.save_output:
                    clear_regression_folder()
                raise

            step += 1
            action_counts[action_name] += 1
            dn = float(torch.norm(output_initial - output_final))
            norms.append(dn)
            parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
            logger.info("step %s | %s | ||Δout||: %s", step, action_name, dn)

            if args.save_output:
                draw_filtered_fx_graph(
                    gm,
                    FOLDER_NAME + "/" + f"fx_graph_simplified{step}",
                    fmt="pdf",
                )
                draw_torch_fx_graph(gm, FOLDER_NAME + "/" + f"fx_graph{step}", fmt="pdf")

    _log_action_summary(action_counts)
    plot_norms_and_parameter_count(norms, parameter_amounts)

    if not args.save_output:
        clear_regression_folder()
