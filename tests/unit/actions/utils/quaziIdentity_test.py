import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.utils import quaziIdentity

VECTOR_SHAPE_TEST = 30

"Square eye_stretch is the identity matrix"
def test_eye_stretch_square_is_identity():
    n = 5
    got = quaziIdentity.eye_stretch(n, n)
    assert got.shape == (n, n)
    assert np.allclose(got, np.eye(n))


"Down-projecting a random vector with a resheper keeps mean and variance close to the original"
def test_resheper_down_project_preserves_mean_and_variance():
    # Arrange
    quaziIdentity.clear_reshepers_cache()
    rng = np.random.default_rng(42)
    n_from, n_to = VECTOR_SHAPE_TEST, int(round(VECTOR_SHAPE_TEST * 0.9))
    x = rng.standard_normal((VECTOR_SHAPE_TEST,1))

    # Act
    R = quaziIdentity.get_reshsper(n_from, n_to)
    y = R.T @ x

    # Assert
    assert y.shape == (n_to, 1)
    print(np.mean(y), np.mean(x))
    print(np.var(y), np.var(x))
    assert np.isclose(np.mean(y), np.mean(x), rtol=0.2, atol=0.15)
    assert np.isclose(np.var(y), np.var(x), rtol=0.35, atol=0.15)


"Round-trip through a ~10% smaller resheper and back recovers the vector approximately"
def test_resheper_round_trip_small_change_matches_original():
    # Arrange
    quaziIdentity.clear_reshepers_cache()
    vector_shape = VECTOR_SHAPE_TEST

    rng = np.random.default_rng(42)
    n_from, n_to = VECTOR_SHAPE_TEST, int(round(VECTOR_SHAPE_TEST* 0.9))
    x = rng.standard_normal((VECTOR_SHAPE_TEST, 1))

    # Act
    R_shrink = quaziIdentity.get_reshsper(n_from, n_to)
    x_shrinked = R_shrink.T @ x
    R_expand = quaziIdentity.get_reshsper(n_to, n_from)
    y = R_expand.T @ x_shrinked

    # Assert
    assert y.shape[0] == vector_shape
    print(np.mean(y), np.mean(x))
    print(np.var(y), np.var(x))
    assert np.isclose(np.mean(y), np.mean(x), rtol=0.2, atol=0.15)


"get_reshsper(n, n-1) differs from the matching identity block by at most 1.0 per entry"
def test_get_reshsper_vs_truncated_identity_max_abs_entrywise_diff_at_most_one():
    # Arrange
    quaziIdentity.clear_reshepers_cache()
    n = VECTOR_SHAPE_TEST
    n_small = VECTOR_SHAPE_TEST - 1
    identity_block = np.eye(n, n_small, dtype=np.float32)

    # Act
    R = quaziIdentity.get_reshsper(n, n_small)
    max_abs = float(np.max(np.abs(R - identity_block)))

    # Assert
    assert R.shape == (n, n_small)
    assert max_abs <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
