"""
Plotting functions for ordination results.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List, Dict, Any, Tuple
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ..ordination.base import OrdinationResult, ConstrainedOrdinationResult


def plot_ordination(
    result: OrdinationResult,
    axes: Tuple[int, int] = (0, 1),
    display: str = "sites",
    choices: Optional[List[int]] = None,
    type: str = "points",
    groups: Optional[Union[np.ndarray, pd.Series]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    scaling: Optional[Union[int, str]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot ordination results.

    Parameters:
        result: OrdinationResult object
        axes: Which axes to plot
        display: What to display ("sites", "species", "both")
        choices: Alternative way to specify axes
        type: Plot type ("points", "text", "none")
        groups: Grouping factor for coloring points
        colors: Colors for groups
        figsize: Figure size
        **kwargs: Additional plotting arguments

    Returns:
        matplotlib Figure object
    """
    if choices is not None:
        axes = tuple(np.array(choices) - 1)  # Convert to 0-based indexing

    fig, ax = plt.subplots(figsize=figsize)

    scaling_id = None
    if hasattr(result, "_normalize_scaling"):
        try:
            scaling_id = result._normalize_scaling(scaling)
        except ValueError:
            scaling_id = None

    scaling_to_use = scaling_id if scaling_id is not None else scaling

    try:
        site_scores, species_scores = result.get_scores(
            display="both", scaling=scaling_to_use
        )
    except (AttributeError, ValueError):
        site_scores = getattr(result, "points", None)
        species_scores = getattr(result, "species", None)
        scaling_id = None

    # Plot sites
    if display in ["sites", "both"]:
        if site_scores is not None:
            points = site_scores
            x = points[:, axes[0]]
            y = points[:, axes[1]]
            labels = [f"Site{i + 1}" for i in range(len(x))]

            if groups is not None:
                _plot_grouped_points(ax, x, y, labels, groups, colors, type, **kwargs)
            else:
                _plot_points(ax, x, y, labels, type, **kwargs)

    # Plot species
    if display in ["species", "both"]:
        if species_scores is not None:
            x_sp = species_scores[:, axes[0]]
            y_sp = species_scores[:, axes[1]]
            labels_sp = [f"Sp{i + 1}" for i in range(len(x_sp))]

            ax.scatter(
                x_sp, y_sp, c="red", marker="^", s=50, alpha=0.7, label="Species"
            )

            if type == "text":
                for i, label in enumerate(labels_sp):
                    ax.annotate(
                        label,
                        (x_sp[i], y_sp[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color="red",
                    )

    # Add biplot arrows for constrained ordination
    if isinstance(result, ConstrainedOrdinationResult):
        biplot_scores = getattr(result, "biplot_scores", None)
        if biplot_scores is not None:
            arrows = np.array(biplot_scores, copy=True)
            if scaling_id in (2, 3):
                try:
                    _, species_mult = result._scaling_multipliers(scaling_id)
                    cols = min(arrows.shape[1], len(species_mult))
                    for idx in range(cols):
                        arrows[:, idx] *= species_mult[idx]
                except (AttributeError, ValueError):
                    pass
            _add_biplot_arrows(ax, arrows, axes, scaling=scaling_id)

    # Set labels
    if hasattr(result, "eigenvalues") and result.eigenvalues is not None:
        if len(result.eigenvalues) > max(axes):
            xlabel = f"Axis {axes[0] + 1} ({result.eigenvalues[axes[0]]:.3f})"
            ylabel = f"Axis {axes[1] + 1} ({result.eigenvalues[axes[1]]:.3f})"
        else:
            xlabel = f"Axis {axes[0] + 1}"
            ylabel = f"Axis {axes[1] + 1}"
    else:
        xlabel = f"Axis {axes[0] + 1}"
        ylabel = f"Axis {axes[1] + 1}"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Add stress information for NMDS
    if hasattr(result, "stress") and result.stress is not None:
        ax.text(
            0.02,
            0.98,
            f"Stress: {result.stress:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    return fig


def ordiplot(
    result: OrdinationResult,
    axes: Tuple[int, int] = (0, 1),
    scaling: Union[str, int] = "species",
    correlation: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    loading_factor: Optional[float] = None,
    predictor_factor: Optional[float] = None,
    axes_source: str = "auto",
    **kwargs,
) -> plt.Figure:
    """
    Create an ordination plot with optional scaling controls for loadings
    and environmental predictors.

    Parameters:
        result: OrdinationResult object
        axes: Which axes to plot
        scaling: Scaling identifier ("sites", "species", "symmetric", 1/2/3)
        correlation: Whether to draw correlation biplot arrows
        figsize: Figure size
        loading_factor: Manual multiplier for species scores. If 0, scales
            loadings to match the longest score vector.
        predictor_factor: Manual multiplier for environmental (predictor)
            vectors in RDA triplots. If 0, scales to the longest score vector.
        axes_source: Which component block to visualise. One of
            ``\"combined\"`` (default), ``\"constrained\"``, or ``\"unconstrained\"``.
            ``\"auto\"`` maps to ``\"combined\"`` for backwards compatibility.
        **kwargs: Additional plotting arguments passed to the site scatter.

    Returns:
        matplotlib Figure object
    """
    axes_source_key = str(axes_source).strip().lower()
    valid_sources = {"auto", "combined", "constrained", "unconstrained"}
    if axes_source_key not in valid_sources:
        raise ValueError(
            "axes_source must be one of 'auto', 'combined', 'constrained', or "
            "'unconstrained'"
        )

    axes_mode = (
        "combined" if axes_source_key in ("auto", "combined") else axes_source_key
    )
    scaling_map = {1: 1, "sites": 1, 2: 2, "species": 2, 3: 3, "symmetric": 3, "sym": 3}
    scaling_id = scaling_map.get(scaling, 2)
    axes_tuple = _ensure_axes_tuple(axes)

    fig, ax = plt.subplots(figsize=figsize)

    try:
        sites, species = result.get_scores(display="both", scaling=scaling_id)
    except (AttributeError, ValueError):
        sites = getattr(result, "points", None)
        species = getattr(result, "species", None)
        scaling_id = None

    site_axis_labels = _extract_axis_labels(sites)
    species_axis_labels = _extract_axis_labels(species)
    site_coords = _scores_to_numpy(sites)
    species_coords = _scores_to_numpy(species)

    total_dims = _max_dimension(site_coords, species_coords)
    if axes_mode in {"constrained", "unconstrained"} and isinstance(
        result, ConstrainedOrdinationResult
    ):
        constrained_dims = int(getattr(result, "rank", 0))
        unconstrained_dims = max(total_dims - constrained_dims, 0)
        if axes_mode == "constrained":
            subset_start = 0
            dims_available = constrained_dims
        else:
            subset_start = constrained_dims
            dims_available = unconstrained_dims

        if dims_available <= 0:
            warnings.warn(
                f"No {axes_mode} axes available; reverting to combined ordiplot.",
                RuntimeWarning,
                stacklevel=2,
            )
            axes_mode = "combined"
        else:
            site_coords = _slice_columns(
                site_coords, subset_start, subset_start + dims_available
            )
            species_coords = _slice_columns(
                species_coords, subset_start, subset_start + dims_available
            )
            site_axis_labels = _slice_labels(
                site_axis_labels, subset_start, subset_start + dims_available
            )
            species_axis_labels = _slice_labels(
                species_axis_labels, subset_start, subset_start + dims_available
            )
            axes_tuple = _normalize_axes_indices(axes_tuple, dims_available)
    else:
        axes_tuple = _normalize_axes_indices(axes_tuple, total_dims)

    site_reference = _max_vector_length(site_coords, axes_tuple)
    if site_reference == 0 and site_coords is None:
        site_reference = _max_vector_length(species_coords, axes_tuple)

    if species_coords is not None and loading_factor is not None:
        species_scale = _resolve_scale_factor(
            loading_factor, species_coords, axes_tuple, site_reference
        )
        species_coords = species_coords * species_scale

    if _coords_support_axes(site_coords, axes_tuple):
        x = site_coords[:, axes_tuple[0]]
        y = site_coords[:, axes_tuple[1]]
        labels = [f"Site{i + 1}" for i in range(len(x))]

        ax.scatter(x, y, alpha=0.7, **kwargs)
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )

    if _coords_support_axes(species_coords, axes_tuple):
        x_sp = species_coords[:, axes_tuple[0]]
        y_sp = species_coords[:, axes_tuple[1]]
        labels_sp = [f"Sp{i + 1}" for i in range(len(x_sp))]

        ax.scatter(x_sp, y_sp, c="red", marker="^", s=50, alpha=0.7, label="Species")
        for i, label in enumerate(labels_sp):
            ax.annotate(
                label,
                (x_sp[i], y_sp[i]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                color="red",
            )

    predictor_reference = max(
        site_reference,
        _max_vector_length(species_coords, axes_tuple)
        if species_coords is not None
        else 0.0,
    )

    if (
        axes_mode != "unconstrained"
        and hasattr(result, "biplot_scores")
        and result.biplot_scores is not None
    ):
        biplot_scores = _scores_to_numpy(result.biplot_scores)
        if biplot_scores is not None and scaling_id in (2, 3):
            try:
                _, species_mult = result._scaling_multipliers(scaling_id)
                cols = min(biplot_scores.shape[1], len(species_mult))
                for idx in range(cols):
                    biplot_scores[:, idx] *= species_mult[idx]
            except (AttributeError, ValueError):
                pass

        if biplot_scores is not None and predictor_factor is not None:
            arrow_scale = _resolve_scale_factor(
                predictor_factor, biplot_scores, axes_tuple, predictor_reference
            )
            biplot_scores = biplot_scores * arrow_scale

        if biplot_scores is not None:
            _add_biplot_arrows(
                ax,
                biplot_scores,
                axes_tuple,
                scaling=scaling_id,
                correlation=correlation,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    xlabel, ylabel = _derive_axis_labels(
        result, axes_mode, axes_tuple, site_axis_labels
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


def biplot(*args, **kwargs) -> plt.Figure:
    warnings.warn(
        "`biplot` has been renamed to `ordiplot`; please update your code.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ordiplot(*args, **kwargs)


def ordiellipse(
    result: OrdinationResult,
    groups: Union[np.ndarray, pd.Series],
    axes: Tuple[int, int] = (0, 1),
    conf: float = 0.95,
    figsize: Tuple[int, int] = (8, 6),
    scaling: Optional[Union[int, str]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Add confidence ellipses to ordination plot.

    Parameters:
        result: OrdinationResult object
        groups: Grouping factor
        axes: Which axes to plot
        conf: Confidence level for ellipses
        figsize: Figure size
        **kwargs: Additional plotting arguments

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot basic ordination
    plot_ordination(result, axes=axes, groups=groups, scaling=scaling, **kwargs)

    # Add ellipses
    if isinstance(groups, pd.Series):
        groups = groups.values

    points = result.get_scores(display="sites", scaling=scaling)
    x = points[:, axes[0]]
    y = points[:, axes[1]]

    unique_groups = np.unique(groups)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

    for i, group in enumerate(unique_groups):
        mask = groups == group
        if np.sum(mask) > 2:  # Need at least 3 points for ellipse
            x_group = x[mask]
            y_group = y[mask]

            # Calculate ellipse parameters
            cov = np.cov(x_group, y_group)
            eigenvals, eigenvecs = np.linalg.eigh(cov)

            # Calculate ellipse parameters
            theta = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = 2 * np.sqrt(eigenvals[0] * chi2.ppf(conf, df=2))
            height = 2 * np.sqrt(eigenvals[1] * chi2.ppf(conf, df=2))

            # Add ellipse
            ellipse = Ellipse(
                xy=(np.mean(x_group), np.mean(y_group)),
                width=width,
                height=height,
                angle=theta,
                facecolor=colors[i],
                alpha=0.3,
                edgecolor=colors[i],
                linewidth=2,
            )
            ax.add_patch(ellipse)

    plt.tight_layout()
    return fig


def ordispider(
    result: OrdinationResult,
    groups: Union[np.ndarray, pd.Series],
    axes: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (8, 6),
    scaling: Optional[Union[int, str]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Add spider plots (lines from centroid to points) to ordination.

    Parameters:
        result: OrdinationResult object
        groups: Grouping factor
        axes: Which axes to plot
        figsize: Figure size
        **kwargs: Additional plotting arguments

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot basic ordination
    plot_ordination(result, axes=axes, groups=groups, scaling=scaling, **kwargs)

    # Add spider lines
    if isinstance(groups, pd.Series):
        groups = groups.values

    points = result.get_scores(display="sites", scaling=scaling)
    x = points[:, axes[0]]
    y = points[:, axes[1]]

    unique_groups = np.unique(groups)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

    for i, group in enumerate(unique_groups):
        mask = groups == group
        x_group = x[mask]
        y_group = y[mask]

        # Calculate centroid
        centroid_x = np.mean(x_group)
        centroid_y = np.mean(y_group)

        # Draw lines from centroid to points
        for j in range(len(x_group)):
            ax.plot(
                [centroid_x, x_group[j]],
                [centroid_y, y_group[j]],
                color=colors[i],
                alpha=0.6,
                linewidth=1,
            )

        # Mark centroid
        ax.scatter(
            centroid_x, centroid_y, color=colors[i], s=100, marker="x", linewidth=3
        )

    plt.tight_layout()
    return fig


def _scores_to_numpy(
    scores: Optional[Union[np.ndarray, pd.DataFrame]],
) -> Optional[np.ndarray]:
    """Convert score containers to a float numpy array."""
    if scores is None:
        return None

    if isinstance(scores, (pd.DataFrame, pd.Series)):
        array = scores.to_numpy()
    else:
        array = np.asarray(scores)

    array = np.array(array, dtype=float, copy=True)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _max_vector_length(coords: Optional[np.ndarray], axes: Tuple[int, int]) -> float:
    """Return the longest vector length within the selected axes."""
    if coords is None:
        return 0.0

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)

    valid_axes = [ax for ax in axes if 0 <= ax < coords.shape[1]]
    if not valid_axes:
        return 0.0

    subset = coords[:, valid_axes]
    if subset.ndim == 1:
        subset = subset.reshape(-1, 1)

    lengths = np.linalg.norm(subset, axis=1)
    return float(np.max(lengths)) if lengths.size else 0.0


def _resolve_scale_factor(
    raw_factor: float,
    coords: np.ndarray,
    axes: Tuple[int, int],
    reference_length: float,
) -> float:
    """Determine the scaling multiplier based on user input."""
    try:
        factor = float(raw_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scale factors must be numeric.") from exc

    if np.isclose(factor, 0.0):
        data_length = _max_vector_length(coords, axes)
        if data_length == 0 or reference_length == 0:
            return 1.0
        return reference_length / data_length

    return factor


def _extract_axis_labels(
    scores: Optional[Union[np.ndarray, pd.DataFrame]],
) -> Optional[List[str]]:
    """Return column labels if available."""
    if isinstance(scores, pd.DataFrame):
        return list(scores.columns)
    return None


def _slice_labels(
    labels: Optional[List[str]],
    start: int,
    stop: int,
) -> Optional[List[str]]:
    if labels is None:
        return None
    start = max(start, 0)
    stop = min(stop, len(labels))
    return labels[start:stop]


def _slice_columns(
    coords: Optional[np.ndarray],
    start: int,
    stop: int,
) -> Optional[np.ndarray]:
    if coords is None:
        return None
    start = max(start, 0)
    stop = max(start, min(stop, coords.shape[1]))
    return coords[:, start:stop]


def _ensure_axes_tuple(axes: Tuple[int, int]) -> Tuple[int, int]:
    """Normalise axes input to a 2-tuple of integers."""
    if len(axes) != 2:
        raise ValueError("axes must be a tuple of two integers")
    return int(axes[0]), int(axes[1])


def _normalize_axes_indices(axes: Tuple[int, int], dims: int) -> Tuple[int, int]:
    """Clamp and extend axes indices to valid bounds."""
    if dims <= 0:
        return (0, 0)
    valid: List[int] = []
    seen = set()
    for ax in axes:
        if 0 <= ax < dims and ax not in seen:
            valid.append(ax)
            seen.add(ax)
    candidate = 0
    while len(valid) < 2 and candidate < dims:
        if candidate not in seen:
            valid.append(candidate)
            seen.add(candidate)
        candidate += 1
    while len(valid) < 2:
        valid.append(0)
    return valid[0], valid[1]


def _coords_support_axes(coords: Optional[np.ndarray], axes: Tuple[int, int]) -> bool:
    """Return True when coords provide both requested axes."""
    if coords is None:
        return False
    max_axis = max(axes)
    return coords.ndim == 2 and coords.shape[1] > max_axis


def _max_dimension(*coords: Optional[np.ndarray]) -> int:
    """Greatest column count across coordinate arrays."""
    dims = [arr.shape[1] for arr in coords if arr is not None and arr.ndim == 2]
    return max(dims) if dims else 0


def _derive_axis_labels(
    result: OrdinationResult,
    axes_mode: str,
    axes: Tuple[int, int],
    axis_labels: Optional[List[str]],
) -> Tuple[str, str]:
    """Build human-readable axis labels including eigenvalues where possible."""

    def format_label(index: int) -> str:
        base = _axis_base_name(axis_labels, axes_mode, index)
        eig = _axis_eigenvalue(result, axes_mode, index)
        if eig is not None:
            return f"{base} ({eig:.3f})"
        return base

    return format_label(axes[0]), format_label(axes[1])


def _axis_base_name(
    labels: Optional[List[str]],
    axes_mode: str,
    index: int,
) -> str:
    if labels is not None and index < len(labels):
        return str(labels[index])
    if axes_mode == "constrained":
        return f"RDA{index + 1}"
    if axes_mode == "unconstrained":
        return f"PC{index + 1}"
    return f"Axis {index + 1}"


def _axis_eigenvalue(
    result: OrdinationResult,
    axes_mode: str,
    index: int,
) -> Optional[float]:
    if isinstance(result, ConstrainedOrdinationResult):
        if axes_mode == "constrained":
            eig = getattr(result, "constrained_eig", None)
        elif axes_mode == "unconstrained":
            eig = getattr(result, "unconstrained_eig", None)
        else:
            eig = getattr(result, "eigenvalues", None)
    else:
        eig = getattr(result, "eigenvalues", None)

    if eig is not None and index < len(eig):
        return float(eig[index])
    return None


# Helper functions
def _plot_points(ax, x, y, labels, type, **kwargs):
    """Plot points on axis."""
    if type == "points":
        ax.scatter(x, y, **kwargs)
    elif type == "text":
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), ha="center", va="center", **kwargs)
    # type == "none" plots nothing


def _plot_grouped_points(ax, x, y, labels, groups, colors, type, **kwargs):
    """Plot grouped points with different colors."""
    if isinstance(groups, pd.Series):
        groups = groups.values

    unique_groups = np.unique(groups)

    if colors is None:
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

    for i, group in enumerate(unique_groups):
        mask = groups == group
        x_group = x[mask] if hasattr(x, "__getitem__") else x.iloc[mask]
        y_group = y[mask] if hasattr(y, "__getitem__") else y.iloc[mask]

        if type == "points":
            ax.scatter(x_group, y_group, color=colors[i], label=group, **kwargs)
        elif type == "text":
            labels_group = [labels[j] for j in range(len(labels)) if mask[j]]
            for j, label in enumerate(labels_group):
                ax.annotate(
                    label,
                    (x_group.iloc[j], y_group.iloc[j]),
                    ha="center",
                    va="center",
                    color=colors[i],
                    **kwargs,
                )

    if type == "points":
        ax.legend()


def _add_biplot_arrows(
    ax, biplot, axes, scaling="species", eig_axis1=1.0, eig_axis2=1.0, correlation=False
):
    """Add biplot arrows to plot.

    Parameters:
        ax: Matplotlib axes object
        biplot: Biplot scores array
        axes: Tuple of axis indices
        scaling: Scaling type ("sites", "species", "symmetric")
        eig_axis1: Eigenvalue for first axis
        eig_axis2: Eigenvalue for second axis
        correlation: Whether to show correlation biplot
    """
    if biplot.shape[1] <= max(axes):
        return

    if isinstance(scaling, int):
        scaling = {1: "sites", 2: "species", 3: "symmetric"}.get(scaling, "species")

    # Get arrow coordinates
    arrow_x = biplot[:, axes[0]]
    arrow_y = biplot[:, axes[1]]

    # Apply scaling to biplot arrows
    # Biplot scores are typically scaled with species in vegan
    if scaling == "sites":
        # When focusing on sites, scale environmental arrows down
        arrow_x = arrow_x / np.sqrt(eig_axis1)
        arrow_y = arrow_y / np.sqrt(eig_axis2)
    elif scaling == "symmetric":
        # Symmetric scaling
        arrow_x = arrow_x / (eig_axis1**0.25)
        arrow_y = arrow_y / (eig_axis2**0.25)
    # For "species" scaling, arrows are already appropriately scaled

    # Scale arrows if needed for visualization
    if not correlation:
        # Scale arrows to fit in plot
        max_coord = max(np.max(np.abs(arrow_x)), np.max(np.abs(arrow_y)))
        if max_coord > 0:
            scale_factor = 0.8 / max_coord
            arrow_x *= scale_factor
            arrow_y *= scale_factor

    # Draw arrows
    for i in range(len(arrow_x)):
        ax.arrow(
            0,
            0,
            arrow_x[i],
            arrow_y[i],
            head_width=0.02,
            head_length=0.03,
            fc="blue",
            ec="blue",
            alpha=0.7,
            linewidth=1.5,
        )

        # Add labels
        ax.annotate(
            f"Var{i + 1}",
            (arrow_x[i], arrow_y[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="red",
            weight="bold",
        )
