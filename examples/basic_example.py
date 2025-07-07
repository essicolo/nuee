#!/usr/bin/env python3
"""
Basic nuee Example
====================

This example demonstrates the core functionality of nuee including:
- Loading sample datasets
- NMDS ordination
- Diversity analysis
- RDA (Redundancy Analysis)
- PERMANOVA
- Visualization

Run this script to see nuee in action!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import nuee 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    """Main example function."""
    print("🌿 nuee Basic Example")
    print("=" * 40)
    
    # Load sample data
    print("\n1. Loading sample data...")
    species_data = nuee.datasets.varespec()
    env_data = nuee.datasets.varechem()
    
    print(f"Species data shape: {species_data.shape}")
    print(f"Environmental data shape: {env_data.shape}")
    print(f"Species data preview:\n{species_data.iloc[:5, :5]}")
    
    # NMDS Ordination
    print("\n2. Running NMDS ordination...")
    nmds_result = nuee.metaMDS(species_data, k=2, distance="bray", trace=True)
    print(f"NMDS Stress: {nmds_result.stress:.4f}")
    print(f"NMDS converged: {nmds_result.converged}")
    
    # Plot NMDS
    print("\n3. Plotting NMDS results...")
    fig1 = nuee.plot_ordination(nmds_result, display="sites", type="points")
    plt.title("NMDS Ordination of Lichen Communities")
    plt.savefig("nmds_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Diversity Analysis
    print("\n4. Calculating diversity indices...")
    shannon_div = nuee.shannon(species_data)
    simpson_div = nuee.simpson(species_data)
    richness = nuee.specnumber(species_data)
    fisher_alpha = nuee.fisher_alpha(species_data)
    
    print(f"Shannon diversity - Mean: {shannon_div.mean():.3f} ± {shannon_div.std():.3f}")
    print(f"Simpson diversity - Mean: {simpson_div.mean():.3f} ± {simpson_div.std():.3f}")
    print(f"Species richness - Mean: {richness.mean():.1f} ± {richness.std():.1f}")
    print(f"Fisher's alpha - Mean: {fisher_alpha.mean():.3f} ± {fisher_alpha.std():.3f}")
    
    # Create diversity comparison plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(shannon_div, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Shannon Diversity")
    axes[0, 0].set_xlabel("Diversity")
    axes[0, 0].set_ylabel("Frequency")
    
    axes[0, 1].hist(simpson_div, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title("Simpson Diversity")
    axes[0, 1].set_xlabel("Diversity")
    axes[0, 1].set_ylabel("Frequency")
    
    axes[1, 0].hist(richness, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title("Species Richness")
    axes[1, 0].set_xlabel("Number of Species")
    axes[1, 0].set_ylabel("Frequency")
    
    axes[1, 1].hist(fisher_alpha, bins=10, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title("Fisher's Alpha")
    axes[1, 1].set_xlabel("Alpha")
    axes[1, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("diversity_histograms.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # RDA Analysis
    print("\n5. Running RDA (Redundancy Analysis)...")
    rda_result = nuee.rda(species_data, env_data, scale=False)
    
    print(f"RDA rank (constrained axes): {rda_result.rank}")
    print(f"Total inertia: {rda_result.tot_chi:.3f}")
    
    if rda_result.constrained_eig is not None:
        print(f"Constrained eigenvalues: {rda_result.constrained_eig[:3]}")
        explained_var = rda_result.explained_variance_ratio[:3]
        print(f"Explained variance ratio: {explained_var}")
    
    # Plot RDA biplot
    print("\n6. Creating RDA biplot...")
    fig3 = nuee.biplot(rda_result)
    plt.title("RDA Biplot: Species ~ Environment")
    plt.savefig("rda_biplot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Distance/Dissimilarity Analysis
    print("\n7. Calculating ecological distances...")
    bray_dist = nuee.vegdist(species_data, method="bray")
    jaccard_dist = nuee.vegdist(species_data, method="jaccard")
    euclidean_dist = nuee.vegdist(species_data, method="euclidean")
    
    print(f"Bray-Curtis distance - Mean: {np.mean(bray_dist[bray_dist > 0]):.3f}")
    print(f"Jaccard distance - Mean: {np.mean(jaccard_dist[jaccard_dist > 0]):.3f}")
    print(f"Euclidean distance - Mean: {np.mean(euclidean_dist[euclidean_dist > 0]):.3f}")
    
    # PERMANOVA
    print("\n8. Running PERMANOVA...")
    try:
        # Create a simple grouping variable based on pH
        ph_groups = pd.cut(env_data['pH'], bins=3, labels=['Low', 'Medium', 'High'])
        
        # Run basic distance-based test
        print(f"Testing effect of pH groups on community composition...")
        print(f"pH groups: {ph_groups.value_counts()}")
        
        # For now, just show the distance matrix summary
        print(f"Bray-Curtis distance matrix summary:")
        print(f"  Min: {np.min(bray_dist[bray_dist > 0]):.3f}")
        print(f"  Max: {np.max(bray_dist):.3f}")
        print(f"  Mean: {np.mean(bray_dist[bray_dist > 0]):.3f}")
        
    except Exception as e:
        print(f"PERMANOVA analysis needs further implementation: {e}")
    
    # Renyi Diversity
    print("\n9. Calculating Renyi diversity...")
    renyi_result = nuee.renyi(species_data, scales=[0, 1, 2, np.inf])
    
    if isinstance(renyi_result, pd.DataFrame):
        print("Renyi diversity (first 5 sites):")
        print(renyi_result.head())
    
    # Summary
    print("\n" + "=" * 40)
    print("🎉 nuee Example Complete!")
    print("=" * 40)
    print("Generated files:")
    print("- nmds_plot.png: NMDS ordination plot")
    print("- diversity_histograms.png: Diversity indices histograms")
    print("- rda_biplot.png: RDA biplot")
    print("\nExplore more nuee features in the documentation!")


if __name__ == "__main__":
    main()