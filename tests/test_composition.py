import warnings

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


# --- ALR denominator pre-fill -----------------------------------------------

def _incomplete_composition(seed=0, n=40, D=5, frac=0.25):
    """Composition where every column has gaps, forcing the pre-fill path."""
    rng = np.random.default_rng(seed)
    truth = np.exp(rng.normal(size=(n, D)))
    mat = truth.copy()
    mask = rng.random((n, D)) < frac
    mask[mask.all(axis=1)] = False
    mat[mask] = np.nan
    return truth, mat, mask


def test_impute_warns_when_no_component_is_complete():
    """The pre-filled denominator is returned unrefined, so say so."""
    _, mat, _ = _incomplete_composition()
    assert np.isnan(mat).sum(axis=0).min() > 0  # no complete column

    with pytest.warns(UserWarning, match="ALR denominator"):
        nuee.impute_missing(mat)


def test_impute_does_not_warn_when_a_component_is_complete():
    rng = np.random.default_rng(2)
    mat = np.exp(rng.normal(size=(30, 4)))
    mask = rng.random((30, 4)) < 0.2
    mask[:, 0] = False              # keep column 0 complete
    mask[mask.all(axis=1)] = False
    mat[mask] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        nuee.impute_missing(mat)


def test_prefilled_denominator_is_as_accurate_as_other_cells():
    """The denominator must not be systematically worse than the cells EM
    actually refines.

    Error is measured in log space, the natural scale for compositions:
    relative errors on lognormal data are heavy-tailed, and their mean is
    dominated by a handful of outliers. The per-replicate ratios are
    summarised with a median for the same reason.
    """
    ratios = []
    for seed in range(40):
        truth, mat, mask = _incomplete_composition(seed=seed)
        na_counts = np.isnan(mat).sum(axis=0)
        if na_counts.min() == 0:
            continue
        denom_idx = int(np.argmin(na_counts))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out = np.asarray(nuee.impute_missing(mat))

        error = np.abs(np.log(out) - np.log(truth))
        denom_cells = mask[:, denom_idx]
        other = mask.copy()
        other[:, denom_idx] = False
        if denom_cells.any() and other.any():
            ratios.append(error[:, denom_idx][denom_cells].mean()
                          / error[other].mean())

    assert len(ratios) > 20, "too few replicates exercised the pre-fill path"
    assert float(np.median(ratios)) < 1.05


def test_impute_preserves_observed_values_without_a_complete_column():
    _, mat, mask = _incomplete_composition(seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = np.asarray(nuee.impute_missing(mat))

    assert np.array_equal(out[~mask], mat[~mask])
    assert not np.isnan(out).any()
    assert (out > 0).all()
