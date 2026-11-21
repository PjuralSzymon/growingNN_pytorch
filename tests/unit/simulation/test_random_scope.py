import random

import numpy as np
import pytest
import torch

from growingnn.utils.random_scope import isolated_search_randomness


def _after_search(draws, fail=False):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    try:
        with isolated_search_randomness():
            for _ in range(draws):
                random.random()
                np.random.random()
                torch.rand(1)
            if fail:
                raise ValueError('search failure')
    except ValueError:
        if not fail:
            raise
    return random.random(), np.random.random(), torch.rand(1).item()


@pytest.mark.parametrize('fail', [False, True])
def test_search_work_does_not_advance_training_rngs(fail):
    assert _after_search(1) == _after_search(20, fail=fail)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_search_restores_cuda_rng():
    state = torch.cuda.get_rng_state()
    expected = torch.rand(4, device='cuda').cpu()
    torch.cuda.set_rng_state(state)
    with isolated_search_randomness():
        torch.rand(100, device='cuda')
    torch.testing.assert_close(torch.rand(4, device='cuda').cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_search_seed_repeats_cuda_draws():
    random.seed(987)
    with isolated_search_randomness():
        first = torch.rand(4, device='cuda').cpu()
    torch.rand(20, device='cuda')
    random.seed(987)
    with isolated_search_randomness():
        second = torch.rand(4, device='cuda').cpu()
    torch.testing.assert_close(first, second)
