"""
Regression: random all-action loop on larger vision models.

Torchvision: ResNet-50, EfficientNet-B2. Hugging Face (pip install transformers): ViT, DeiT, ConvNeXt, MobileNetV2.
"""

import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B2_Weights,
    ResNet50_Weights,
    efficientnet_b2,
    resnet50,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_layer import AddSeqLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.logger import logger
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    log_regression_action_error,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)

BATCH_SIZE_VISION = 4
INPUT_SHAPE = (3, 64, 64)
BATCH_SIZE_HF_VISION = 2
HF_INPUT_SHAPE = (3, 224, 224)
ITERATIONS = 10
SEED = 42
OUT_ROOT = FOLDER_NAME + "/big_models"

ModelSpec = Tuple[str, Callable[[], nn.Module], Callable[[torch.Generator], torch.Tensor]]


class _HfVisionWrapper(nn.Module):
    """Trace-friendly wrapper: pixel_values in, last_hidden_state out."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state


class _HfViTWrapper(nn.Module):
    """HF ViT with trace-friendly forward (skip HF guards and shape unpacking)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n_batch = pixel_values.size(0)
        emb = self.model.embeddings
        pe = emb.patch_embeddings
        hidden = pe.projection(pixel_values).flatten(2).transpose(1, 2)
        hidden = torch.cat((emb.cls_token.expand(n_batch, -1, -1), hidden), dim=1)
        hidden = emb.dropout(hidden + emb.position_embeddings)
        for layer in self.model.layers:
            hidden = _hf_vit_encoder_layer(layer, hidden, n_batch)
        return self.model.layernorm(hidden)


def _hf_vit_encoder_layer(layer: nn.Module, hidden: torch.Tensor, n_batch: int) -> torch.Tensor:
    """Single ViT encoder block without hidden_states.shape unpacking."""
    attn = layer.attention
    n_heads, head_dim = attn.num_attention_heads, attn.head_dim
    qkv_shape = (n_batch, -1, n_heads, head_dim)

    def mha(x: torch.Tensor) -> torch.Tensor:
        q = attn.q_proj(x).view(qkv_shape).transpose(1, 2)
        k = attn.k_proj(x).view(qkv_shape).transpose(1, 2)
        v = attn.v_proj(x).view(qkv_shape).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scaling
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(n_batch, -1, n_heads * head_dim)
        return attn.o_proj(out)

    residual = hidden
    hidden = layer.layernorm_before(hidden)
    hidden = layer.dropout(mha(hidden)) + residual
    residual = hidden
    hidden = layer.layernorm_after(hidden)
    hidden = layer.dropout(layer.mlp(hidden)) + residual
    return hidden


def _vision_x(rng: torch.Generator) -> torch.Tensor:
    return torch.randn(BATCH_SIZE_VISION, *INPUT_SHAPE, generator=rng)


def _hf_vision_x(rng: torch.Generator) -> torch.Tensor:
    return torch.randn(BATCH_SIZE_HF_VISION, *HF_INPUT_SHAPE, generator=rng)


def _load_resnet50() -> nn.Module:
    return resnet50(weights=ResNet50_Weights.DEFAULT).eval()


def _load_efficientnet_b2() -> nn.Module:
    return efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT).eval()


def _load_mobilenet_v2_hf() -> nn.Module:
    from transformers import MobileNetV2Config, MobileNetV2Model
    config = MobileNetV2Config.from_pretrained("google/mobilenet_v2_1.0_224")
    config.tf_padding = False  # TF SAME padding uses int(shape); breaks FX trace
    model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224", config=config)
    return _HfVisionWrapper(model.eval())


def _load_vit() -> nn.Module:
    from transformers import ViTModel
    return _HfViTWrapper(ViTModel.from_pretrained("google/vit-base-patch16-224").eval())

MODEL_SPECS: List[ModelSpec] = [
    ("resnet50", _load_resnet50, _vision_x),
    ("efficientnet_b2", _load_efficientnet_b2, _vision_x),
    ("mobilenet_v2_hf", _load_mobilenet_v2_hf, _hf_vision_x),
    ("vit-base-patch16-224", _load_vit, _hf_vision_x),
]

ActionGenerator = Tuple[str, Callable[[fx.GraphModule], List[Action]]]

ACTION_GENERATORS: List[ActionGenerator] = [
    ("AddResLayer", lambda gm: AddResLayer.generate_all_actions(gm, layer_types=[Layer_Type.EYE])),
    ("AddResConvLayer", AddResConvLayer.generate_all_actions),
    ("AddSeqLayer", AddSeqLayer.generate_all_actions),
    ("AddSeqConvLayer", AddSeqConvLayer.generate_all_actions),
    ("DelLayer", DelLayer.generate_all_actions),
    ("DelNeurons", DelNeurons.generate_all_actions),
]


def _log_action_summary(model_name: str, action_counts: dict[str, int]) -> None:
    total = sum(action_counts.values())
    logger.info("[%s] action summary (%d total):", model_name, total)
    col = max((len(name) for name in action_counts), default=6)
    logger.info("%-*s | %s", col, "action", "count")
    logger.info("%s-+-%s", "-" * col, "-" * 5)
    for name in sorted(action_counts):
        logger.info("%-*s | %d", col, name, action_counts[name])


def _run_model(name: str, gm: fx.GraphModule, x: torch.Tensor, args, rng: random.Random) -> None:
    with torch.no_grad():
        output_initial = gm(x)

    norms: List[float] = []
    parameter_amounts: List[int] = [GraphStructureQuery.get_amount_of_parameters(gm)]
    action_counts: dict[str, int] = {n: 0 for n, _ in ACTION_GENERATORS}
    out_dir = f"{OUT_ROOT}/{name}"

    if args.save_output:
        draw_filtered_fx_graph(gm, out_dir + "/fx_graph_simplified0", fmt="pdf")
        draw_torch_fx_graph(gm, out_dir + "/fx_graph0", fmt="pdf")
    logger.info("[%s] initial graph ready (%d params)", name, parameter_amounts[0])

    step = 0
    for iteration in range(ITERATIONS):
        logger.info("[%s] iteration: %s --------------------------------", name, iteration)
        order = list(ACTION_GENERATORS)
        rng.shuffle(order)

        for action_name, generate in order:
            actions = generate(gm)
            if not actions:
                continue

            idx = rng.randrange(len(actions))
            chosen = actions[idx]
            logger.info(
                "[%s] iteration %s | %s | picked %s/%s: %s",
                name, iteration, action_name, idx, len(actions), chosen,
            )
            try:
                chosen.execute(gm)
                with torch.no_grad():
                    output_final = gm(x)
            except Exception:
                if args.save_output:
                    draw_filtered_fx_graph(
                        gm,
                        out_dir + f"/fx_graph_simplified_error_iter{iteration}_{action_name}",
                        fmt="pdf",
                    )
                log_regression_action_error(
                    gm, chosen, actions=actions, action_type=action_name,
                    norms=norms, parameter_amounts=parameter_amounts, action_counts=action_counts,
                    model=name,
                )
                _log_action_summary(name, action_counts)
                plot_norms_and_parameter_count(
                    norms, parameter_amounts, save_path=out_dir + "/norms_and_params.png"
                )
                raise

            step += 1
            action_counts[action_name] += 1
            dn = float(torch.norm(output_initial - output_final))
            norms.append(dn)
            parameter_amounts.append(GraphStructureQuery.get_amount_of_parameters(gm))
            logger.info("[%s] step %s | %s | ||Δout||: %s", name, step, action_name, dn)

            if args.save_output:
                draw_filtered_fx_graph(gm, out_dir + f"/fx_graph_simplified{step}", fmt="pdf")
                draw_torch_fx_graph(gm, out_dir + f"/fx_graph{step}", fmt="pdf")

    _log_action_summary(name, action_counts)
    plot_norms_and_parameter_count(
        norms, parameter_amounts, save_path=out_dir + "/norms_and_params.png"
    )


if __name__ == "__main__":
    args = parse_regression_cli()
    rng = random.Random(SEED)
    data_rng = torch.Generator().manual_seed(SEED)

    for name, load_model, make_x in MODEL_SPECS:
        logger.info("======== model: %s ========", name)
        model = load_model()
        x = make_x(data_rng)
        gm = fx.symbolic_trace(model)
        with torch.no_grad():
            gm(x)
        _run_model(name, gm, x, args, rng)

    if not args.save_output:
        clear_regression_folder()
