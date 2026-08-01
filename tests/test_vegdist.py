import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

import nuee


ALL_METHODS = [
    "bray", "jaccard", "euclidean", "manhattan", "canberra", "gower",
    "altGower", "morisita", "horn", "mountford", "raup", "binomial",
    "chao", "cao", "kulczynski", "mahalanobis",
]


@pytest.fixture(scope="module")
def species():
    return nuee.datasets.varespec()


def _condensed(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return squareform(matrix, checks=False) if matrix.ndim == 2 else matrix


# --- general properties -------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_method_returns_a_valid_dissimilarity(method, species):
    matrix = np.asarray(nuee.vegdist(species, method=method), dtype=float)

    assert matrix.shape == (species.shape[0], species.shape[0])
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.all(np.isfinite(matrix))


# Raup-Crick is a probability, not a metric: identical sites still leave a
# small chance of sharing fewer species by draw, so it does not reach zero.
# vegan reports 0.004545455 for the pair below, and so does nuee.
ZERO_ON_IDENTITY = [m for m in ALL_METHODS if m != "raup"]


@pytest.mark.parametrize("method", ZERO_ON_IDENTITY)
def test_identical_rows_are_at_zero_distance(method):
    rng = np.random.default_rng(0)
    row = rng.integers(0, 10, size=12).astype(float)
    data = np.vstack([row, row, rng.integers(1, 10, size=12).astype(float)])

    matrix = np.asarray(nuee.vegdist(data, method=method), dtype=float)
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_raup_is_a_probability_not_a_metric():
    row = np.array([3, 7, 0, 9, 2, 0, 5, 1, 8, 4, 0, 6], dtype=float)
    data = np.vstack([row, row, np.arange(1, 13, dtype=float)])

    matrix = np.asarray(nuee.vegdist(data, method="raup"), dtype=float)
    assert matrix[0, 1] == pytest.approx(0.004545455, abs=1e-8)


def test_unknown_method_is_rejected(species):
    with pytest.raises(ValueError, match="Unknown distance method"):
        nuee.vegdist(species, method="not-a-method")


# --- indices whose definition was wrong --------------------------------------

def test_jaccard_uses_abundances_not_presence(species):
    """Regression: jaccard returned the binary index, silently discarding
    abundances. vegan's quantitative form is 2B/(1+B)."""
    bray = _condensed(nuee.vegdist(species, method="bray"))
    jaccard = _condensed(nuee.vegdist(species, method="jaccard"))

    np.testing.assert_allclose(jaccard, 2 * bray / (1 + bray), atol=1e-12)

    binary = pdist((species.values > 0).astype(float), metric="jaccard")
    assert not np.allclose(jaccard, binary)


def test_canberra_is_normalised(species):
    """Regression: the raw scipy sum was returned, so values grew with the
    size of the species pool instead of staying in [0, 1]."""
    canberra = _condensed(nuee.vegdist(species, method="canberra"))
    assert canberra.max() <= 1.0
    assert canberra.min() >= 0.0


def test_raup_is_probabilistic_not_jaccard(species):
    """Regression: _raup_crick was `return _jaccard(x)`."""
    raup = _condensed(nuee.vegdist(species, method="raup"))
    jaccard = _condensed(nuee.vegdist(species, method="jaccard"))

    assert not np.allclose(raup, jaccard)
    assert raup.min() >= 0.0 and raup.max() <= 1.0


def test_mountford_stays_within_its_range(species):
    """The Mountford index is bounded by log(2)."""
    mountford = _condensed(nuee.vegdist(species, method="mountford"))
    assert mountford.min() >= 0.0
    assert mountford.max() <= np.log(2.0) + 1e-12


def test_mountford_is_zero_for_a_subset():
    subset = np.array([[1, 1, 1, 1, 1, 0],
                       [1, 1, 0, 0, 0, 0]], dtype=float)
    matrix = np.asarray(nuee.vegdist(subset, method="mountford"), dtype=float)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_mountford_is_log2_for_disjoint_sites():
    disjoint = np.array([[1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1]], dtype=float)
    matrix = np.asarray(nuee.vegdist(disjoint, method="mountford"), dtype=float)
    assert matrix[0, 1] == pytest.approx(np.log(2.0))


def test_morisita_is_never_negative(species):
    """Rounding could push a near-identical pair below zero."""
    counts = species.round().astype(int)
    morisita = _condensed(nuee.vegdist(counts, method="morisita"))
    assert morisita.min() >= 0.0


def test_cao_floors_abundances_rather_than_only_zeros():
    """A value below 0.1 must be lifted to 0.1, not left alone: vegan floors,
    and on non-integer input the two choices differ."""
    data = np.array([[0.55, 2.0, 0.0, 3.0],
                     [0.05, 0.0, 4.0, 3.0]])
    matrix = np.asarray(nuee.vegdist(data, method="cao"), dtype=float)
    assert matrix[0, 1] == pytest.approx(1.51389, abs=1e-5)


def test_gower_uses_the_column_range(species):
    """Gower standardises by each species' range across sites."""
    gower = _condensed(nuee.vegdist(species, method="gower"))
    assert gower.min() >= 0.0
    assert gower.max() <= 1.0


def test_kulczynski_matches_its_definition(species):
    values = species.values.astype(float)
    expected = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            shared = np.minimum(values[i], values[j]).sum()
            expected.append(1 - 0.5 * (shared / values[i].sum()
                                       + shared / values[j].sum()))
    np.testing.assert_allclose(_condensed(nuee.vegdist(species, method="kulczynski")),
                               expected, atol=1e-12)


# --- input handling -----------------------------------------------------------

@pytest.mark.parametrize("method", ["bray", "jaccard", "canberra", "raup"])
def test_dataframe_and_array_agree(method, species):
    from_frame = np.asarray(nuee.vegdist(species, method=method), dtype=float)
    from_array = np.asarray(nuee.vegdist(species.values, method=method), dtype=float)
    np.testing.assert_allclose(from_frame, from_array, atol=1e-12)
