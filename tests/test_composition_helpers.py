"""Property-based checks for the compositional helpers.

These have no vegan counterpart, so they are pinned by the algebraic
identities that define them rather than by reference values.
"""

import numpy as np
import pandas as pd
import pytest

import nuee


@pytest.fixture
def composition():
    rng = np.random.default_rng(0)
    return nuee.closure(np.exp(rng.normal(size=(12, 5))))


# --- alr / alr_inv -----------------------------------------------------------

def test_alr_roundtrip(composition):
    coords = nuee.alr(composition)
    assert coords.shape == (composition.shape[0], composition.shape[1] - 1)
    np.testing.assert_allclose(nuee.alr_inv(coords), composition, atol=1e-10)


@pytest.mark.parametrize("denominator", [0, 2, 4, -1, -2])
def test_alr_roundtrip_for_any_denominator(composition, denominator):
    """Regression: alr_inv passed a negative index straight to np.insert,
    which counts from the end of the *input*, so the denominator landed one
    place short and the default -1 never round-tripped."""
    coords = nuee.alr(composition, denominator_idx=denominator)
    restored = nuee.alr_inv(coords, denominator_idx=denominator)
    np.testing.assert_allclose(restored, composition, atol=1e-10)


def test_alr_is_scale_invariant(composition):
    """ALR reads ratios, so rescaling a row must not move its coordinates."""
    scaled = composition * np.array([[1.0], [5.0]] * 6)
    np.testing.assert_allclose(nuee.alr(scaled), nuee.alr(composition), atol=1e-10)


def test_alr_matches_its_definition(composition):
    expected = np.log(composition[:, :-1] / composition[:, [-1]])
    np.testing.assert_allclose(nuee.alr(composition), expected, atol=1e-12)


# --- clr / ilr consistency ---------------------------------------------------

def test_clr_rows_sum_to_zero(composition):
    np.testing.assert_allclose(nuee.clr(composition).sum(axis=1), 0.0, atol=1e-10)


def test_ilr_preserves_aitchison_distances(composition):
    """The ilr basis is orthonormal, so it is an isometry."""
    coords = nuee.ilr(composition)
    clr_coords = nuee.clr(composition)

    def pairwise(matrix):
        diff = matrix[:, None, :] - matrix[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    np.testing.assert_allclose(pairwise(coords), pairwise(clr_coords), atol=1e-10)


# --- sbp_basis ---------------------------------------------------------------

def test_sbp_basis_is_orthonormal():
    sbp = np.array([[1, -1, 0], [1, 1, -1]])
    basis = nuee.sbp_basis(sbp)

    gram = basis @ basis.T
    np.testing.assert_allclose(gram, np.eye(basis.shape[0]), atol=1e-10)


def test_sbp_basis_rows_sum_to_zero():
    basis = nuee.sbp_basis(np.array([[1, -1, 0], [1, 1, -1]]))
    np.testing.assert_allclose(basis.sum(axis=1), 0.0, atol=1e-10)


# --- perturbation / power ----------------------------------------------------

def test_power_by_one_is_the_identity(composition):
    np.testing.assert_allclose(nuee.power(composition, 1.0), composition, atol=1e-10)


def test_power_result_stays_a_composition(composition):
    powered = nuee.power(composition, 2.5)
    np.testing.assert_allclose(powered.sum(axis=1), 1.0, atol=1e-10)


def test_power_is_linear_in_clr_space(composition):
    """Powering scales clr coordinates by the same factor."""
    np.testing.assert_allclose(nuee.clr(nuee.power(composition, 3.0)),
                               3.0 * nuee.clr(composition), atol=1e-9)


# --- centre / centralize -----------------------------------------------------

def _closed_geometric_mean(matrix):
    centre = np.exp(np.log(matrix).mean(axis=0))
    return centre / centre.sum()


def test_center_is_an_alias_for_centralize(composition):
    """`center` divides by the compositional centre; it does not return it."""
    np.testing.assert_allclose(nuee.center(composition),
                               nuee.centralize(composition), atol=1e-12)


def test_centralize_moves_the_centre_to_the_barycentre(composition):
    centred = np.asarray(nuee.centralize(composition))
    n_parts = composition.shape[1]
    np.testing.assert_allclose(_closed_geometric_mean(centred),
                               np.full(n_parts, 1.0 / n_parts), atol=1e-9)


def test_centralize_output_is_a_composition(composition):
    np.testing.assert_allclose(np.asarray(nuee.centralize(composition)).sum(axis=1),
                               1.0, atol=1e-10)


# --- replace_zeros -----------------------------------------------------------

def test_replace_zeros_preserves_row_sums():
    data = np.array([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [2.0, 2.0, 2.0]])
    replaced = nuee.replace_zeros(data)

    np.testing.assert_allclose(replaced.sum(axis=1), data.sum(axis=1), atol=1e-10)
    assert (replaced > 0).all()


def test_replace_zeros_leaves_zero_free_rows_untouched():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]])
    replaced = nuee.replace_zeros(data)
    np.testing.assert_array_equal(replaced[0], data[0])


def test_replace_zeros_honours_detection_limits():
    data = np.array([[1.0, 0.0, 2.0]])
    limits = np.array([0.5, 0.5, 0.5])
    replaced = nuee.replace_zeros(data, detection_limits=limits, delta=0.65)
    assert replaced[0, 1] == pytest.approx(0.65 * 0.5)


def test_replace_zeros_rejects_negative_values():
    with pytest.raises(ValueError, match="negative"):
        nuee.replace_zeros(np.array([[1.0, -1.0]]))


def test_replace_zeros_rejects_an_all_zero_row():
    with pytest.raises(ValueError, match="entirely zeros"):
        nuee.replace_zeros(np.array([[1.0, 2.0], [0.0, 0.0]]))


def test_replace_zeros_preserves_dataframe_labels():
    frame = pd.DataFrame([[1.0, 0.0, 2.0]], columns=list("abc"), index=["s1"])
    replaced = nuee.replace_zeros(frame)
    assert isinstance(replaced, pd.DataFrame)
    assert list(replaced.columns) == list("abc")
    assert list(replaced.index) == ["s1"]


# --- inner --------------------------------------------------------------------

def test_inner_matches_the_pairwise_log_ratio_definition(composition):
    """Regression: inner divided the clr product by D a second time, so every
    entry came out D times too small."""
    n_parts = composition.shape[1]

    def aitchison(x, y):
        total = sum(np.log(x[i] / x[j]) * np.log(y[i] / y[j])
                    for i in range(n_parts) for j in range(i + 1, n_parts))
        return total / n_parts

    expected = np.array([[aitchison(a, b) for b in composition] for a in composition])
    np.testing.assert_allclose(nuee.inner(composition), expected, atol=1e-10)


def test_inner_is_the_clr_gram_matrix(composition):
    """The Aitchison inner product is the ordinary one on clr coordinates."""
    clr_coords = nuee.clr(composition)
    np.testing.assert_allclose(nuee.inner(composition),
                               clr_coords @ clr_coords.T, atol=1e-10)


def test_inner_is_symmetric_with_squared_norms_on_the_diagonal(composition):
    gram = np.asarray(nuee.inner(composition))
    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(gram),
                               (nuee.clr(composition) ** 2).sum(axis=1),
                               atol=1e-10)
