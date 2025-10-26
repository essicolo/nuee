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
    
    # Plot sites
    if display in ["sites", "both"]:
        if hasattr(result, 'points') and result.points is not None:
            points = result.points
            if isinstance(points, pd.DataFrame):
                x = points.iloc[:, axes[0]]
                y = points.iloc[:, axes[1]]
                labels = points.index
            else:
                x = points[:, axes[0]]
                y = points[:, axes[1]]
                labels = [f"Site{i+1}" for i in range(len(x))]
            
            if groups is not None:
                _plot_grouped_points(ax, x, y, labels, groups, colors, type, **kwargs)
            else:
                _plot_points(ax, x, y, labels, type, **kwargs)
    
    # Plot species
    if display in ["species", "both"]:
        if hasattr(result, 'species') and result.species is not None:
            species = result.species
            if isinstance(species, pd.DataFrame):
                x_sp = species.iloc[:, axes[0]]
                y_sp = species.iloc[:, axes[1]]
                labels_sp = species.index
            else:
                x_sp = species[:, axes[0]]
                y_sp = species[:, axes[1]]
                labels_sp = [f"Sp{i+1}" for i in range(len(x_sp))]
            
            # Plot species as red triangles
            ax.scatter(x_sp, y_sp, c='red', marker='^', s=50, alpha=0.7, label='Species')
            
            if type == "text":
                for i, label in enumerate(labels_sp):
                    ax.annotate(label, (x_sp[i], y_sp[i]), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, color='red')
    
    # Add biplot arrows for constrained ordination
    if isinstance(result, ConstrainedOrdinationResult) and result.biplot is not None:
        _add_biplot_arrows(ax, result.biplot, axes)
    
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

    Parameters:
        result: ConstrainedOrdinationResult object
        axes: Which axes to plot
        scaling: Type of scaling (1/"sites", 2/"species", 3/"symmetric")
                 - 1/"sites": Focus on sites (distances between sites meaningful)
                 - 2/"species": Focus on species (distances between species meaningful)
                 - 3/"symmetric": Symmetric scaling of both sites and species
        correlation: Whether to show correlation biplot
        figsize: Figure size
        **kwargs: Additional plotting arguments

    Returns:
        matplotlib Figure object
    """
    # Normalize scaling parameter
    scaling_map = {
        1: "sites", "sites": "sites",
        2: "species", "species": "species",
        3: "symmetric", "symmetric": "symmetric"
    }
    scaling = scaling_map.get(scaling, "species")

    fig, ax = plt.subplots(figsize=figsize)

    # Get eigenvalues for scaling
    if hasattr(result, 'eigenvalues') and result.eigenvalues is not None:
        eigenvalues = result.eigenvalues
    else:
        eigenvalues = np.ones(result.ndim)

    # Calculate scaling factors based on eigenvalues
    if len(eigenvalues) > max(axes):
        eig_axis1 = eigenvalues[axes[0]]
        eig_axis2 = eigenvalues[axes[1]]
    else:
        eig_axis1 = eig_axis2 = 1.0

    # Plot sites with appropriate scaling
    if hasattr(result, 'points') and result.points is not None:
        points = result.points
        if isinstance(points, pd.DataFrame):
            x = points.iloc[:, axes[0]].values
            y = points.iloc[:, axes[1]].values
            labels = points.index
        else:
            x = points[:, axes[0]]
            y = points[:, axes[1]]
            labels = [f"Site{i+1}" for i in range(len(x))]

        # Apply scaling to site scores
        if scaling == "sites":
            # Scale sites by sqrt(eigenvalue) for type 1 scaling
            x = x * np.sqrt(eig_axis1)
            y = y * np.sqrt(eig_axis2)
        elif scaling == "symmetric":
            # Symmetric scaling: scale by eigenvalue^(1/4)
            x = x * (eig_axis1 ** 0.25)
            y = y * (eig_axis2 ** 0.25)
        # For "species" scaling, use raw site scores (divided by sqrt(eig) implicitly)

        ax.scatter(x, y, alpha=0.7, **kwargs)

        # Add site labels
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), xytext=(3, 3),
                       textcoords='offset points', fontsize=8)

    # Plot species if available
    if hasattr(result, 'species') and result.species is not None:
        species = result.species
        if isinstance(species, pd.DataFrame):
            x_sp = species.iloc[:, axes[0]].values
            y_sp = species.iloc[:, axes[1]].values
            labels_sp = species.index
        else:
            x_sp = species[:, axes[0]]
            y_sp = species[:, axes[1]]
            labels_sp = [f"Sp{i+1}" for i in range(len(x_sp))]

        # Apply scaling to species scores (opposite of sites)
        if scaling == "species":
            # Scale species by sqrt(eigenvalue) for type 2 scaling
            x_sp = x_sp * np.sqrt(eig_axis1)
            y_sp = y_sp * np.sqrt(eig_axis2)
        elif scaling == "symmetric":
            # Symmetric scaling
            x_sp = x_sp * (eig_axis1 ** 0.25)
            y_sp = y_sp * (eig_axis2 ** 0.25)

        # Plot species as red triangles
        ax.scatter(x_sp, y_sp, c='red', marker='^', s=50, alpha=0.7, label='Species')
        for i, label in enumerate(labels_sp):
            ax.annotate(label, (x_sp[i], y_sp[i]), xytext=(3, 3),
                       textcoords='offset points', fontsize=8, color='red')

    # Add biplot arrows (environmental variables)
    if hasattr(result, 'biplot_scores') and result.biplot_scores is not None:
        _add_biplot_arrows(ax, result.biplot_scores, axes, scaling, eig_axis1, eig_axis2, correlation)

    # Set equal aspect ratio for biplots
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # Add axis labels
    ax.set_xlabel(f'Axis {axes[0]+1}')
    ax.set_ylabel(f'Axis {axes[1]+1}')

    plt.tight_layout()
    return fig


def ordiplot(result: OrdinationResult,
             axes: Tuple[int, int] = (0, 1),
             display: str = "sites",
             figsize: Tuple[int, int] = (8, 6),
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
                          figsize=figsize, **kwargs)


def ordiellipse(result: OrdinationResult,
                groups: Union[np.ndarray, pd.Series],
                axes: Tuple[int, int] = (0, 1),
                conf: float = 0.95,
                figsize: Tuple[int, int] = (8, 6),
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
    plot_ordination(result, axes=axes, groups=groups, **kwargs)
    
    # Add ellipses
    if isinstance(groups, pd.Series):
        groups = groups.values
    
    points = result.points
    if isinstance(points, pd.DataFrame):
        x = points.iloc[:, axes[0]].values
        y = points.iloc[:, axes[1]].values
    else:
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
    plot_ordination(result, axes=axes, groups=groups, **kwargs)
    
    # Add spider lines
    if isinstance(groups, pd.Series):
        groups = groups.values
    
    points = result.points
    if isinstance(points, pd.DataFrame):
        x = points.iloc[:, axes[0]].values
        y = points.iloc[:, axes[1]].values
    else:
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