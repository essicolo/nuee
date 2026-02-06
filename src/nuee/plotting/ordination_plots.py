"""
Plotting functions for ordination results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List, Dict, Any, Tuple
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ..ordination.base import OrdinationResult, ConstrainedOrdinationResult


def _to_1d(v) -> np.ndarray:
    """Coerce any vector-like input to a 1-D numpy array.

    Handles: Polars DataFrame/Series, pandas DataFrame/Series,
    numpy arrays (1-D or single-column 2-D), and plain lists.
    """
    # Polars DataFrame → first column
    if hasattr(v, "to_series"):        # pl.DataFrame
        v = v.to_series()
    if hasattr(v, "to_numpy"):         # pl.Series or pd.Series/DataFrame
        v = v.to_numpy()
    arr = np.asarray(v)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def plot_ordination(result: OrdinationResult,
                    axes: Tuple[int, int] = (0, 1),
                    display: str = "sites",
                    choices: Optional[List[int]] = None,
                    type: str = "points",
                    groups: Optional[Union[np.ndarray, pd.Series]] = None,
                    colors: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (8, 6),
                    scaling: Optional[Union[int, str]] = None,
                    title: Optional[str] = None,
                    **kwargs) -> plt.Figure:
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
        site_scores, species_scores = result.get_scores(display="both", scaling=scaling_to_use)
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
            labels = [f"Site{i+1}" for i in range(len(x))]

            if groups is not None:
                _plot_grouped_points(ax, x, y, labels, groups, colors, type, **kwargs)
            else:
                _plot_points(ax, x, y, labels, type, **kwargs)
    
    # Plot species
    if display in ["species", "both"]:
        if species_scores is not None:
            x_sp = species_scores[:, axes[0]]
            y_sp = species_scores[:, axes[1]]
            labels_sp = [f"Sp{i+1}" for i in range(len(x_sp))]

            ax.scatter(x_sp, y_sp, c='red', marker='^', s=50, alpha=0.7, label='Species')

            if type == "text":
                for i, label in enumerate(labels_sp):
                    ax.annotate(label, (x_sp[i], y_sp[i]), xytext=(5, 5),
                              textcoords='offset points', fontsize=8, color='red')
    
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
    if hasattr(result, 'eigenvalues') and result.eigenvalues is not None:
        if len(result.eigenvalues) > max(axes):
            xlabel = f"Axis {axes[0]+1} ({result.eigenvalues[axes[0]]:.3f})"
            ylabel = f"Axis {axes[1]+1} ({result.eigenvalues[axes[1]]:.3f})"
        else:
            xlabel = f"Axis {axes[0]+1}"
            ylabel = f"Axis {axes[1]+1}"
    else:
        xlabel = f"Axis {axes[0]+1}"
        ylabel = f"Axis {axes[1]+1}"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # Add stress information for NMDS
    if hasattr(result, 'stress') and result.stress is not None:
        ax.text(0.02, 0.98, f"Stress: {result.stress:.3f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig


def biplot(result: 'OrdinationResult',
           axes: Tuple[int, int] = (0, 1),
           scaling: Union[str, int] = "species",
           correlation: bool = False,
           figsize: Tuple[int, int] = (10, 8),
           title: Optional[str] = None,
           arrow_mul: Optional[float] = None,
           n_species: Optional[int] = 15,
           show_site_labels: bool = True,
           show_species_labels: bool = True,
           repel: bool = True,
           fontsize: int = 8,
           site_kw: Optional[Dict[str, Any]] = None,
           species_kw: Optional[Dict[str, Any]] = None,
           env_kw: Optional[Dict[str, Any]] = None,
           groups: Optional[Union[np.ndarray, pd.Series, list]] = None,
           color_by: Optional[Union[np.ndarray, pd.Series, list]] = None,
           cmap: Optional[str] = None,
           **kwargs) -> plt.Figure:
    """
    Create a biplot for ordination results.

    For unconstrained ordination (PCA, CA, LDA), species loadings are drawn
    as arrows from the origin.  For constrained ordination (RDA / CCA),
    species are shown as points and environmental variables as arrows.

    Parameters
    ----------
    result : OrdinationResult
        Ordination result object.
    axes : tuple of int
        Which ordination axes to plot (0-indexed).
    scaling : str or int
        Scaling mode: 1/"sites", 2/"species", 3/"symmetric".
    correlation : bool
        If True, use raw correlation values without auto-scaling.
    figsize : tuple of int
        Figure size in inches.
    title : str, optional
        Plot title.
    arrow_mul : float, optional
        Manual multiplier for arrow length.
    n_species : int or None
        Show only the top *n_species* by loading magnitude.
        ``None`` shows all species.
    show_site_labels : bool
        Whether to display site name labels.
    show_species_labels : bool
        Whether to display species name labels.
    repel : bool
        Use adjustText for ggrepel-style label placement.
    fontsize : int
        Base font size for labels.
    site_kw : dict, optional
        Extra keyword arguments for site scatter points.
    species_kw : dict, optional
        Extra keyword arguments for species scatter/arrows.
    env_kw : dict, optional
        Extra keyword arguments for environmental arrows.
    groups : array-like, optional
        Categorical group labels (one per site) for coloured scatter.
        Auto-detected from LDA results.
    color_by : array-like, optional
        Continuous values (one per site) for colour-mapped scatter with
        a colourbar.  Mutually exclusive with *groups*.
    cmap : str, optional
        Matplotlib colormap name.  Default is the colour cycle for
        *groups* (up to 10) and ``"viridis"`` for *color_by*.
    **kwargs
        Additional keyword arguments passed to site scatter.
    """
    try:
        from adjustText import adjust_text
        _has_adjusttext = True
    except ImportError:
        _has_adjusttext = False

    # --- Resolve groups / color_by ---
    if groups is None:
        groups = getattr(result, "groups", None)
    if groups is not None and color_by is not None:
        raise ValueError(
            "'groups' and 'color_by' are mutually exclusive; "
            "provide one or neither."
        )
    if groups is not None:
        groups = _to_1d(groups)
    if color_by is not None:
        color_by = _to_1d(color_by).astype(float)

    scaling_map = {
        1: 1, "sites": 1,
        2: 2, "species": 2,
        3: 3, "symmetric": 3, "sym": 3
    }
    scaling_id = scaling_map.get(scaling, 2)

    fig, ax = plt.subplots(figsize=figsize)

    is_constrained = isinstance(result, ConstrainedOrdinationResult)

    try:
        sites, species = result.get_scores(display="both", scaling=scaling_id)
    except (AttributeError, ValueError):
        sites = getattr(result, "points", None)
        species = getattr(result, "species", None)
        scaling_id = None

    if isinstance(sites, pd.DataFrame):
        sites = sites.values
    if isinstance(species, pd.DataFrame):
        species = species.values

    # Collect text objects for repel
    _texts = []

    # Labels
    site_names = getattr(result, "_site_names", None)
    species_names = getattr(result, "_species_names", None)

    # Default site styling — hollow circles so labels read clearly
    _site_kw = dict(s=45, facecolors="none", edgecolors="steelblue",
                    linewidths=1.0, zorder=3, label="Sites")
    _site_kw.update(kwargs)
    if site_kw:
        _site_kw.update(site_kw)

    # --- Plot sites ---
    if sites is not None:
        x = sites[:, axes[0]]
        y = sites[:, axes[1]]
        labels = (site_names if site_names and len(site_names) == len(x)
                  else [f"Site{i+1}" for i in range(len(x))])

        if groups is not None:
            # -- Categorical group coloring --
            unique_groups = list(dict.fromkeys(groups))  # preserve order
            n_groups = len(unique_groups)
            if cmap is not None:
                _cmap = plt.get_cmap(cmap)
            elif n_groups > 10:
                _cmap = plt.get_cmap("tab20")
            else:
                _cmap = None  # use default colour cycle

            for idx, grp in enumerate(unique_groups):
                mask = groups == grp
                color = f"C{idx}" if _cmap is None else _cmap(
                    idx / max(n_groups - 1, 1))
                grp_kw = dict(_site_kw)
                grp_kw.pop("facecolors", None)
                grp_kw.pop("edgecolors", None)
                grp_kw.pop("label", None)
                grp_kw["color"] = color
                grp_kw["label"] = str(grp)
                ax.scatter(x[mask], y[mask], **grp_kw)

                if show_site_labels:
                    for i in np.where(mask)[0]:
                        t = ax.text(x[i], y[i], labels[i],
                                    fontsize=fontsize, color=color,
                                    alpha=0.75, zorder=4)
                        _texts.append(t)

        elif color_by is not None:
            # -- Continuous coloring --
            cont_kw = dict(_site_kw)
            cont_kw.pop("facecolors", None)
            cont_kw.pop("edgecolors", None)
            cont_kw.pop("label", None)
            cont_kw["edgecolors"] = "face"
            sc = ax.scatter(x, y, c=color_by, cmap=cmap or "viridis",
                            **cont_kw)
            plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

            if show_site_labels:
                for i, label in enumerate(labels):
                    t = ax.text(x[i], y[i], label, fontsize=fontsize,
                                alpha=0.75, zorder=4)
                    _texts.append(t)

        else:
            # -- Default: single scatter (unchanged) --
            ax.scatter(x, y, **_site_kw)
            if show_site_labels:
                for i, label in enumerate(labels):
                    t = ax.text(x[i], y[i], label, fontsize=fontsize,
                                alpha=0.75, zorder=4)
                    _texts.append(t)

    # --- Plot species ---
    if species is not None:
        x_sp = species[:, axes[0]]
        y_sp = species[:, axes[1]]
        n_sp = len(x_sp)
        labels_sp = (species_names
                     if species_names and len(species_names) == n_sp
                     else [f"Sp{i+1}" for i in range(n_sp)])

        # Filter to top n_species by loading magnitude
        magnitudes = np.sqrt(x_sp ** 2 + y_sp ** 2)
        if n_species is not None and n_species < n_sp:
            top_idx = np.argsort(magnitudes)[-n_species:]
        else:
            top_idx = np.arange(n_sp)

        if is_constrained:
            # Constrained ordination: species as points (triangles)
            _sp_kw = dict(c="dimgray", marker="^", s=30, alpha=0.6,
                          zorder=2, label="Species")
            if species_kw:
                _sp_kw.update(species_kw)
            ax.scatter(x_sp[top_idx], y_sp[top_idx], **_sp_kw)
            if show_species_labels:
                for i in top_idx:
                    t = ax.text(x_sp[i], y_sp[i], labels_sp[i],
                                fontsize=fontsize - 1, color="dimgray",
                                alpha=0.8, zorder=4)
                    _texts.append(t)
        else:
            # Unconstrained ordination (PCA / CA): species as arrows
            _arrow_color = "dimgray"
            if species_kw and "color" in species_kw:
                _arrow_color = species_kw["color"]
            for i in top_idx:
                ax.annotate(
                    "", xy=(x_sp[i], y_sp[i]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=_arrow_color,
                                    lw=1.2, alpha=0.6),
                    zorder=2)
                if show_species_labels:
                    t = ax.text(x_sp[i], y_sp[i], labels_sp[i],
                                fontsize=fontsize - 1, color=_arrow_color,
                                fontweight="bold", alpha=0.8, zorder=4)
                    _texts.append(t)

    # --- Biplot arrows for env variables (constrained ordination) ---
    env_names = None
    call = getattr(result, "call", {})
    if isinstance(call, dict):
        env_names = call.get("constraints", None)

    if hasattr(result, "biplot_scores") and result.biplot_scores is not None:
        biplot_scores = np.array(result.biplot_scores, copy=True)
        if scaling_id in (2, 3):
            try:
                _, species_mult = result._scaling_multipliers(scaling_id)
                cols = min(biplot_scores.shape[1], len(species_mult))
                for idx in range(cols):
                    biplot_scores[:, idx] *= species_mult[idx]
            except (AttributeError, ValueError):
                pass

        data_extent = 1.0
        all_coords = []
        if sites is not None:
            all_coords.append(sites[:, axes[0]])
            all_coords.append(sites[:, axes[1]])
        if species is not None:
            all_coords.append(species[:, axes[0]])
            all_coords.append(species[:, axes[1]])
        if all_coords:
            data_extent = max(np.max(np.abs(np.concatenate(all_coords))),
                              1e-12)

        env_texts = _add_biplot_arrows(
            ax, biplot_scores, axes,
            scaling=scaling_id,
            correlation=correlation,
            data_extent=data_extent,
            arrow_mul=arrow_mul,
            var_names=env_names,
            fontsize=fontsize,
            env_kw=env_kw,
        )
        _texts.extend(env_texts)

    # --- Repel labels ---
    if repel and _has_adjusttext and _texts:
        adjust_text(_texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                   lw=0.4, alpha=0.6))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Axis labels — method-specific
    _method = (result.call.get("method", "")
               if isinstance(getattr(result, "call", None), dict) else "")
    if _method == "LDA":
        _evar = getattr(result, "explained_variance_ratio_", None)
        if _evar is not None and len(_evar) > max(axes):
            ax.set_xlabel(f"LD{axes[0]+1} ({_evar[axes[0]]:.1%})")
            ax.set_ylabel(f"LD{axes[1]+1} ({_evar[axes[1]]:.1%})")
        else:
            ax.set_xlabel(f"LD{axes[0]+1}")
            ax.set_ylabel(f"LD{axes[1]+1}")
    else:
        ax.set_xlabel(f"Axis {axes[0]+1}")
        ax.set_ylabel(f"Axis {axes[1]+1}")

    if title is not None:
        ax.set_title(title)

    # Legend
    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fontsize, loc="best", framealpha=0.7)

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_nmds(result, axes=(0, 1), figsize=(10, 8), title=None,
              show_species=True, show_site_labels=True,
              show_species_labels=True, n_species=15,
              repel=True, fontsize=8, **kwargs):
    """Plot NMDS ordination (sites and optionally species scores).

    Parameters
    ----------
    result : NMDSResult
        NMDS ordination result.
    axes : tuple of int
        Which axes to display.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    show_species : bool
        Overlay species weighted-average scores.
    show_site_labels, show_species_labels : bool
        Toggle labels for each element type.
    n_species : int or None
        Limit to top N species by distance from origin.
    repel : bool
        Use adjustText for label repulsion.
    fontsize : int
        Base font size.
    **kwargs
        Extra keyword arguments for site scatter.
    """
    try:
        from adjustText import adjust_text
        _has_adjusttext = True
    except ImportError:
        _has_adjusttext = False

    fig, ax = plt.subplots(figsize=figsize)
    _texts = []

    # --- Sites ---
    sites = getattr(result, "points", None)
    if sites is not None:
        if isinstance(sites, pd.DataFrame):
            site_names = list(sites.index)
            sites = sites.values
        else:
            site_names = getattr(result, "_site_names", None)
            if not site_names:
                site_names = [f"Site{i+1}" for i in range(sites.shape[0])]

        x = sites[:, axes[0]]
        y = sites[:, axes[1]]

        _site_kw = dict(s=45, facecolors="none", edgecolors="steelblue",
                        linewidths=1.0, zorder=3, label="Sites")
        _site_kw.update(kwargs)
        ax.scatter(x, y, **_site_kw)

        if show_site_labels:
            for i, name in enumerate(site_names):
                t = ax.text(x[i], y[i], str(name), fontsize=fontsize,
                            color="steelblue", alpha=0.8, zorder=4)
                _texts.append(t)

    # --- Species (weighted averages) ---
    species = getattr(result, "species", None)
    if show_species and species is not None:
        if isinstance(species, pd.DataFrame):
            sp_names = list(species.index)
            species = species.values
        else:
            sp_names = getattr(result, "_species_names", None)
            if not sp_names:
                sp_names = [f"Sp{i+1}" for i in range(species.shape[0])]

        x_sp = species[:, axes[0]]
        y_sp = species[:, axes[1]]
        n_sp = len(x_sp)

        magnitudes = np.sqrt(x_sp ** 2 + y_sp ** 2)
        if n_species is not None and n_species < n_sp:
            top_idx = np.argsort(magnitudes)[-n_species:]
        else:
            top_idx = np.arange(n_sp)

        ax.scatter(x_sp[top_idx], y_sp[top_idx], c="dimgray", marker="+",
                   s=40, alpha=0.6, linewidths=0.8, zorder=2,
                   label="Species")

        if show_species_labels:
            for i in top_idx:
                t = ax.text(x_sp[i], y_sp[i], sp_names[i],
                            fontsize=fontsize - 1, color="dimgray",
                            alpha=0.8, zorder=4)
                _texts.append(t)

    # --- Repel ---
    if repel and _has_adjusttext and _texts:
        adjust_text(_texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                   lw=0.4, alpha=0.6))

    # --- Stress annotation ---
    stress = getattr(result, "stress", None)
    if stress is not None:
        ax.text(0.02, 0.98, f"Stress: {stress:.3f}",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=fontsize,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel(f"NMDS{axes[0]+1}")
    ax.set_ylabel(f"NMDS{axes[1]+1}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    if title is not None:
        ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fontsize, loc="best", framealpha=0.7)

    plt.tight_layout()
    plt.close(fig)
    return fig


def ordiplot(result: OrdinationResult,
             axes: Tuple[int, int] = (0, 1),
             display: str = "sites",
             figsize: Tuple[int, int] = (8, 6),
             scaling: Optional[Union[int, str]] = None,
             **kwargs) -> plt.Figure:
    """
    Basic ordination plot.
    
    Parameters:
        result: OrdinationResult object
        axes: Which axes to plot
        display: What to display ("sites", "species", "both")
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    return plot_ordination(result, axes=axes, display=display,
                          figsize=figsize, scaling=scaling, **kwargs)


def ordiellipse(result: OrdinationResult,
                groups: Union[np.ndarray, pd.Series],
                axes: Tuple[int, int] = (0, 1),
                conf: float = 0.95,
                figsize: Tuple[int, int] = (8, 6),
                scaling: Optional[Union[int, str]] = None,
                **kwargs) -> plt.Figure:
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
            ellipse = Ellipse(xy=(np.mean(x_group), np.mean(y_group)),
                            width=width, height=height, angle=theta,
                            facecolor=colors[i], alpha=0.3, 
                            edgecolor=colors[i], linewidth=2)
            ax.add_patch(ellipse)
    
    plt.tight_layout()
    return fig


def ordispider(result: OrdinationResult,
               groups: Union[np.ndarray, pd.Series],
               axes: Tuple[int, int] = (0, 1),
               figsize: Tuple[int, int] = (8, 6),
               scaling: Optional[Union[int, str]] = None,
               **kwargs) -> plt.Figure:
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
            ax.plot([centroid_x, x_group[j]], [centroid_y, y_group[j]], 
                   color=colors[i], alpha=0.6, linewidth=1)
        
        # Mark centroid
        ax.scatter(centroid_x, centroid_y, color=colors[i], 
                  s=100, marker='x', linewidth=3)
    
    plt.tight_layout()
    return fig


# Helper functions
def _plot_points(ax, x, y, labels, type, **kwargs):
    """Plot points on axis."""
    if type == "points":
        ax.scatter(x, y, **kwargs)
    elif type == "text":
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), ha='center', va='center', **kwargs)
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
        x_group = x[mask] if hasattr(x, '__getitem__') else x.iloc[mask]
        y_group = y[mask] if hasattr(y, '__getitem__') else y.iloc[mask]
        
        if type == "points":
            ax.scatter(x_group, y_group, color=colors[i], label=group, **kwargs)
        elif type == "text":
            labels_group = [labels[j] for j in range(len(labels)) if mask[j]]
            for j, label in enumerate(labels_group):
                ax.annotate(label, (x_group.iloc[j], y_group.iloc[j]), 
                          ha='center', va='center', color=colors[i], **kwargs)
    
    if type == "points":
        ax.legend()


def _add_biplot_arrows(ax, biplot, axes, scaling="species", eig_axis1=1.0,
                       eig_axis2=1.0, correlation=False,
                       data_extent=1.0, arrow_mul=None,
                       var_names=None, fontsize=8, env_kw=None):
    """Add biplot arrows to plot and return text objects.

    Parameters
    ----------
    ax : matplotlib Axes
    biplot : ndarray
        Biplot scores array.
    axes : tuple of int
        Axis indices.
    scaling : str or int
    eig_axis1, eig_axis2 : float
        Eigenvalues for the selected axes.
    correlation : bool
    data_extent : float
        Maximum absolute coordinate of plotted data.
    arrow_mul : float, optional
    var_names : list of str, optional
        Environmental variable names.
    fontsize : int
    env_kw : dict, optional
        Extra styling for arrows.

    Returns
    -------
    list of matplotlib Text
        Text objects for the arrow labels (for repel).
    """
    texts = []
    if biplot.shape[1] <= max(axes):
        return texts

    if isinstance(scaling, int):
        scaling = {1: "sites", 2: "species", 3: "symmetric"}.get(scaling, "species")

    arrow_x = biplot[:, axes[0]].copy()
    arrow_y = biplot[:, axes[1]].copy()

    if scaling == "sites":
        arrow_x = arrow_x / np.sqrt(eig_axis1)
        arrow_y = arrow_y / np.sqrt(eig_axis2)
    elif scaling == "symmetric":
        arrow_x = arrow_x / (eig_axis1 ** 0.25)
        arrow_y = arrow_y / (eig_axis2 ** 0.25)

    if not correlation:
        max_coord = max(np.max(np.abs(arrow_x)), np.max(np.abs(arrow_y)))
        if max_coord > 0:
            scale_factor = (0.8 * data_extent) / max_coord
            arrow_x = arrow_x * scale_factor
            arrow_y = arrow_y * scale_factor

    if arrow_mul is not None:
        arrow_x = arrow_x * arrow_mul
        arrow_y = arrow_y * arrow_mul

    _color = "steelblue"
    _lw = 1.5
    _alpha = 0.8
    if env_kw:
        _color = env_kw.get("color", _color)
        _lw = env_kw.get("lw", _lw)
        _alpha = env_kw.get("alpha", _alpha)

    if var_names is None or len(var_names) != len(arrow_x):
        var_names = [f"Var{i+1}" for i in range(len(arrow_x))]

    for i in range(len(arrow_x)):
        ax.annotate(
            "", xy=(arrow_x[i], arrow_y[i]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=_color,
                            lw=_lw, alpha=_alpha,
                            mutation_scale=12),
            zorder=5)
        t = ax.text(arrow_x[i], arrow_y[i], var_names[i],
                    fontsize=fontsize, color=_color, fontweight="bold",
                    alpha=0.9, zorder=6)
        texts.append(t)

    # Add a single invisible handle for legend
    ax.plot([], [], color=_color, lw=_lw, label="Env. variables")

    return texts
