import numpy as np
import pytest

import nuee


def test_closure_rows_sum_to_one():
    data = np.array([[2.0, 3.0, 5.0], [4.0, 0.0, 6.0]])
    closed = nuee.closure(data)
    np.testing.assert_allclose(closed.sum(axis=1), np.ones(2))


def test_multiplicative_replacement_eliminates_zeros():
    data = np.array([[0.0, 5.0, 10.0]])
    replaced = nuee.multiplicative_replacement(data)
    assert np.all(replaced > 0)
    np.testing.assert_allclose(replaced.sum(), 1.0)


def test_clr_inversion_round_trip():
    data = np.array([[0.2, 0.3, 0.5]])
    coords = nuee.clr(data)
    recovered = nuee.clr_inv(coords)
    np.testing.assert_allclose(recovered, data)


def test_ilr_round_trip_matches_original():
    data = np.array([[0.25, 0.25, 0.25, 0.25]])
    coords = nuee.ilr(data)
    recovered = nuee.ilr_inv(coords)
    np.testing.assert_allclose(recovered, data)


def test_ilr_requires_positive_entries():
    with pytest.raises(ValueError):
        nuee.ilr(np.array([[0.0, 1.0, 0.0]]))
