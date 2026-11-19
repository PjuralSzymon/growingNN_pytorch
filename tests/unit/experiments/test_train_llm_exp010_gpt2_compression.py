import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.train_llm_exp010_gpt2_compression import (
    _benchmark_output_agreement,
    _benchmark_prompts,
    _generation_agreement,
    _make_prompts,
)
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.trainer import train_generations
from growingnn.training.causal_lm import CausalLMNextTokenAdapter, pack_token_inputs


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 16)
        self.received_mask = None
        self.received_positions = None

    def forward(self, *, input_ids, attention_mask, position_ids, use_cache, return_dict):
        self.received_mask = attention_mask.detach().clone()
        self.received_positions = position_ids.detach().clone()
        values = self.embedding(input_ids)
        values = values * attention_mask.unsqueeze(-1)
        return (self.projection(values),)


def test_prompt_partitions_are_reproducible_and_disjoint():
    train_a, val_a = _make_prompts(12, 5, seed=42)
    train_b, val_b = _make_prompts(12, 5, seed=42)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(set(train_a)) == len(train_a)
    assert len(set(val_a)) == len(val_a)
    assert set(train_a).isdisjoint(val_a)


def test_benchmark_prompts_are_repeatable_and_outside_training_and_validation():
    train, validation = _make_prompts(12, 5, seed=42)
    a = _benchmark_prompts(train + validation, count=10, seed=99)
    b = _benchmark_prompts(train + validation, count=10, seed=99)
    assert a == b
    assert len(set(a)) == 10
    assert set(a).isdisjoint(train + validation)


def test_same_prompt_benchmark_compares_generated_tokens():
    before = {"outputs": [
        {"generated_token_ids": [1, 2, 3]},
        {"generated_token_ids": [4, 5]},
    ]}
    after = {"outputs": [
        {"generated_token_ids": [1, 7, 3]},
        {"generated_token_ids": [4, 5]},
    ]}
    result = _benchmark_output_agreement(before, after)
    assert result == {
        "matched_tokens": 4, "total_tokens": 5, "token_accuracy": 0.8,
        "exact_sequences": 1, "total_sequences": 2, "exact_sequence_fraction": 0.5,
    }


def test_next_token_training_preserves_and_passes_padding_mask():
    ids = torch.tensor([[0, 2, 3], [4, 5, 6]])
    masks = torch.tensor([[0, 1, 1], [1, 1, 1]])
    packed = pack_token_inputs(ids, masks)
    targets = torch.tensor([3, 6])
    teacher = TinyCausalLM()
    model = CausalLMNextTokenAdapter(teacher)
    loader = DataLoader(TensorDataset(packed, targets), batch_size=2)

    _, history = gradient_descent(
        model, 1, loader, loader, nn.CrossEntropyLoss(),
        ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        quiet=True, device="cpu",
    )

    assert history["train_loss"]
    assert teacher.received_mask.tolist() == masks.tolist()
    assert teacher.received_positions.tolist() == [[0, 0, 1], [0, 1, 2]]


def test_heldout_greedy_generation_agreement_counts_tokens_and_sequences():
    class AlwaysSeven(nn.Module):
        def forward(self, packed):
            logits = torch.zeros((packed.size(0), 10), device=packed.device)
            logits[:, 7] = 1
            return logits

    result = _generation_agreement(
        AlwaysSeven(),
        [{"prompt_token_ids": [1, 2], "teacher_token_ids": [7, 7]},
         {"prompt_token_ids": [3], "teacher_token_ids": [7, 4]}],
        context=4, pad_token_id=0, device=torch.device("cpu"),
    )
    assert result["token_accuracy"] == 0.75
    assert result["exact_sequences"] == 1
    assert result["total_sequences"] == 2


def test_gpt2_width_action_reduces_parameters_and_preserves_output_width():
    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(8, 20)
            self.left = nn.Linear(20, 20)
            self.right = nn.Linear(20, 20)
            self.output = nn.Linear(20, 4)

        def forward(self, x):
            hidden = self.stem(x)
            return self.output(self.left(hidden) + self.right(hidden))

    sample = torch.randn(2, 8)
    traced = TracedModel.create(SmallMLP(), tuple(sample.shape))
    traced.shapes()
    before = traced.param_count()
    actions = DelNeurons.generate_all_actions(traced, ratio=0.75)
    assert actions
    actions[0].execute(traced)
    after = sum(parameter.numel() for parameter in traced.gm.parameters())
    assert after < before
    assert traced.gm(sample).shape == (2, 4)


def test_generation_callback_receives_model_and_validation_history():
    model = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    observed = []
    post_actions = []
    config = RunningConfig(
        generations=1, epochs=1,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        criterion=nn.CrossEntropyLoss(), quiet=True, device="cpu",
    )

    train_generations(
        model, loader, loader, config,
        generation_callback=lambda current, generation, history:
            observed.append((generation, current, history["val_acc"][-1]),
        ),
        post_action_callback=lambda current, generation, action:
            post_actions.append((generation, current, action)),
    )

    assert len(observed) == 1
    assert observed[0][0] == 0
    assert isinstance(observed[0][1], torch.fx.GraphModule)
    assert 0.0 <= observed[0][2] <= 1.0
    assert len(post_actions) == 1
    assert post_actions[0][0] == 0
    assert post_actions[0][2] is None
