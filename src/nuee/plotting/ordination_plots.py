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


def plot_ordination(result: OrdinationResult,
                    axes: Tuple[int, int] = (0, 1),
                    display: str = "sites",
                    choices: Optional[List[int]] = None,
                    type: str = "points",
                    groups: Optional[Union[np.ndarray, pd.Series]] = None,
                    colors: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (8, 6),
                    scaling: Optional[Union[int, str]] = None,
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
    
    plt.tight_layout()
    return fig


def biplot(result: ConstrainedOrdinationResult,
           axes: Tuple[int, int] = (0, 1),
           scaling: Union[str, int] = "species",
           correlation: bool = False,
           figsize: Tuple[int, int] = (8, 6),
           **kwargs) -> plt.Figure:
    """
    Create a biplot for constrained ordination results.
    """
    scaling_map = {
        1: 1, "sites": 1,
        2: 2, "species": 2,
        3: 3, "symmetric": 3, "sym": 3
    }
    scaling_id = scaling_map.get(scaling, 2)

    fig, ax = plt.subplots(figsize=figsize)

    try:
        sites, species = result.get_scores(display="both", scaling=scaling_id)
    except (AttributeError, ValueError):
        sites = getattr(result, "points", None)
        species = getattr(result, "species", None)
        scaling_id = None

    if sites is not None:
        x = sites[:, axes[0]]
        y = sites[:, axes[1]]
        labels = [f"Site{i+1}" for i in range(len(x))]

        ax.scatter(x, y, alpha=0.7, **kwargs)
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), xytext=(3, 3),
                       textcoords='offset points', fontsize=8)

    if species is not None:
        x_sp = species[:, axes[0]]
        y_sp = species[:, axes[1]]
        labels_sp = [f"Sp{i+1}" for i in range(len(x_sp))]

        ax.scatter(x_sp, y_sp, c='red', marker='^', s=50, alpha=0.7, label='Species')
        for i, label in enumerate(labels_sp):
            ax.annotate(label, (x_sp[i], y_sp[i]), xytext=(3, 3),
                       textcoords='offset points', fontsize=8, color='red')

    if hasattr(result, 'biplot_scores') and result.biplot_scores is not None:
        biplot_scores = np.array(result.biplot_scores, copy=True)
        if scaling_id in (2, 3):
            try:
                _, species_mult = result._scaling_multipliers(scaling_id)
                cols = min(biplot_scores.shape[1], len(species_mult))
                for idx in range(cols):
                    biplot_scores[:, idx] *= species_mult[idx]
            except (AttributeError, ValueError):
                pass
        _add_biplot_arrows(ax, biplot_scores, axes,
                           scaling=scaling_id,
                           correlation=correlation)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(f'Axis {axes[0]+1}')
    ax.set_ylabel(f'Axis {axes[1]+1}')
    plt.tight_layout()
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


def _add_biplot_arrows(ax, biplot, axes, scaling="species", eig_axis1=1.0, eig_axis2=1.0, correlation=False):
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
        arrow_x = arrow_x / (eig_axis1 ** 0.25)
        arrow_y = arrow_y / (eig_axis2 ** 0.25)
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
        ax.arrow(0, 0, arrow_x[i], arrow_y[i],
                head_width=0.02, head_length=0.03,
                fc='blue', ec='blue', alpha=0.7, linewidth=1.5)

        # Add labels
        ax.annotate(f'Var{i+1}', (arrow_x[i], arrow_y[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='red', weight='bold')
