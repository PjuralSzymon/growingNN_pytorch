import pytest
import torch

from experiments.train_llm_exp010_self_distillation import (
    distillation_loss,
    make_prompts,
)


def test_exp010_prompts_are_diverse_and_seed_reproducible():
    first = make_prompts(32, seed=10)
    second = make_prompts(32, seed=10)

    assert first == second
    assert len(set(first)) > 8


def test_exp010_distillation_loss_masks_padding_and_backpropagates():
    torch.manual_seed(4)
    teacher_logits = torch.randn(2, 4, 11)
    student_logits = teacher_logits.clone().requires_grad_()
    input_ids = torch.tensor([[1, 2, 3, 0], [2, 4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    loss, cross_entropy, kl = distillation_loss(
        student_logits,
        teacher_logits,
        input_ids,
        attention_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(cross_entropy)
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_exp010_rejects_invalid_distillation_temperature():
    logits = torch.randn(1, 3, 5)
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(tokens)

    with pytest.raises(ValueError, match="temperature must be positive"):
        distillation_loss(
            logits,
            logits,
            tokens,
            mask,
            temperature=0,
        )
