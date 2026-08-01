"""
Plotting functions for dissimilarity analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional


def plot_dissimilarity(distance_matrix: Union[np.ndarray, pd.DataFrame],
                       figsize: tuple = (8, 6),
                       **kwargs) -> plt.Figure:
    """
    Plot dissimilarity matrix as heatmap.
    
    Parameters:
        distance_matrix: Distance/dissimilarity matrix
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(distance_matrix, pd.DataFrame):
        sns.heatmap(distance_matrix, ax=ax, **kwargs)
    else:
        sns.heatmap(distance_matrix, ax=ax, **kwargs)
    
    ax.set_title('Dissimilarity Matrix')
    plt.tight_layout()
    return fig


def plot_betadisper(betadisper_result: dict,
                    figsize: tuple = (8, 6),
                    **kwargs) -> plt.Figure:
    """
    Plot beta dispersion results.
    
    Draws one box per group of the distances to the group centroid, the same
    view as vegan's ``boxplot.betadisper``, with the group means marked and
    the permutation test reported in the title.

    Parameters:
        betadisper_result: Beta dispersion results, from :func:`nuee.betadisper`
        figsize: Figure size
        **kwargs: Additional arguments passed to ``Axes.boxplot``

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: if the result carries no per-observation grouping.
    """
    distances = np.asarray(betadisper_result["distances"], dtype=float)
    groups = betadisper_result.get("groups")
    if groups is None:
        raise ValueError(
            "betadisper_result carries no 'groups'; re-run nuee.betadisper() "
            "to obtain a result that records the grouping."
        )
    groups = np.asarray(groups)

    labels = list(dict.fromkeys(groups.tolist()))
    grouped = [distances[groups == label] for label in labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(grouped, tick_labels=[str(label) for label in labels], **kwargs)

    means = betadisper_result.get("group_means")
    if means is not None:
        ordered = [float(means[label]) for label in labels]
        ax.scatter(range(1, len(labels) + 1), ordered, marker="D",
                   color="firebrick", zorder=3, label="Group mean")
        ax.legend(frameon=False, loc="best")

    ax.set_ylabel("Distance to centroid")
    ax.set_xlabel("Group")

    title = "Beta dispersion"
    f_stat = betadisper_result.get("f_statistic")
    p_value = betadisper_result.get("p_value")
    if f_stat is not None and np.isfinite(f_stat):
        title += f"  (F = {f_stat:.3f}"
        if p_value is not None and np.isfinite(p_value):
            title += f", p = {p_value:.3f}"
        title += ")"
    ax.set_title(title)

    plt.tight_layout()
    return fig