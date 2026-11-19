"""Small adapters for running causal language models through GrowingNN training.

GrowingNN's existing trainer expects a tensor of class logits and a tensor of
class targets. This adapter turns fixed-width causal-LM contexts into next-token
class logits while preserving padding masks and mask-derived position IDs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CausalLMNextTokenAdapter(nn.Module):
    """Adapt packed ``(input_ids, attention_mask)`` batches to next-token logits.

    The packed input has shape ``(batch, 2, sequence)``. Channel zero contains
    token IDs and channel one contains an attention mask. Float storage is used
    because the legacy GrowingNN trainer converts model inputs to float before
    calling the model; IDs and masks are restored to integers here.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 3 or packed.shape[1] != 2:
            raise ValueError(
                "causal LM input must have shape (batch, 2, sequence) with IDs and mask channels"
            )
        input_ids = packed[:, 0, :].long()
        attention_mask = packed[:, 1, :].long()
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        return logits[:, -1, :]


def pack_token_inputs(input_ids: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
    """Pack equal-shaped token IDs and masks for the existing float-input trainer."""
    if input_ids.shape != attention_masks.shape:
        raise ValueError("input_ids and attention_masks must have identical shapes")
    return torch.stack((input_ids.float(), attention_masks.float()), dim=1)


def teacher_token_agreement(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> float:
    """Top-1 agreement with teacher next-token targets from a fixed loader."""
    model.eval()
    correct = total = 0
    device = torch.device(device)
    with torch.inference_mode():
        for inputs, targets in loader:
            predictions = model(inputs.to(device).float()).argmax(dim=1)
            targets = targets.to(device)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return correct / total if total else 0.0
