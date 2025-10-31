import numpy as np
import pandas as pd

import nuee


def _manual_shannon(data, base=np.e):
    totals = data.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    p = data / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)
    log_p[np.isneginf(log_p)] = 0.0
    ent = -(p * log_p).sum(axis=1)
    return ent / np.log(base)


def _manual_simpson(data):
    totals = data.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    p = data / totals
    dominance = (p**2).sum(axis=1)
    return 1.0 - dominance


def _manual_inverse_simpson(data):
    totals = data.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    p = data / totals
    dominance = (p**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dominance
    inv[~np.isfinite(inv)] = 0.0
    return inv


def _manual_richness(data):
    return (data > 0).sum(axis=1)


def test_diversity_indices_match_manual_numpy():
    data = np.array([[2, 1, 0], [0, 3, 4]], dtype=float)

    shannon = nuee.shannon(data).values
    simpson = nuee.simpson(data).values
    invsimpson = nuee.diversity(data, index="invsimpson")
    richness = nuee.specnumber(data).values

    np.testing.assert_allclose(shannon, _manual_shannon(data))
    np.testing.assert_allclose(simpson, _manual_simpson(data))
    np.testing.assert_allclose(invsimpson, _manual_inverse_simpson(data))
    np.testing.assert_array_equal(richness, _manual_richness(data))


def test_diversity_indices_match_manual_dataframe():
    df = pd.DataFrame([[5, 0, 1], [2, 2, 2]], index=["A", "B"])

    shannon = nuee.shannon(df)
    simpson = nuee.simpson(df)
    invsimpson = nuee.diversity(df, index="invsimpson")
    richness = nuee.specnumber(df)

    expected_shannon = _manual_shannon(df.to_numpy())
    expected_simpson = _manual_simpson(df.to_numpy())
    expected_inverse = _manual_inverse_simpson(df.to_numpy())
    expected_richness = _manual_richness(df.to_numpy())

    np.testing.assert_allclose(shannon.values, expected_shannon)
    np.testing.assert_allclose(simpson.values, expected_simpson)
    np.testing.assert_allclose(invsimpson.values, expected_inverse)
    np.testing.assert_array_equal(richness.values, expected_richness)

    assert shannon.sample_names == ["A", "B"]
    assert simpson.sample_names == ["A", "B"]
    assert richness.sample_names == ["A", "B"]
