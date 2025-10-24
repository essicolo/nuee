import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import nuee
    return (nuee,)


@app.cell
def _(nuee):
    species_data = nuee.datasets.varespec()
    env_data = nuee.datasets.varechem()
    return env_data, species_data


@app.cell
def _(nuee, species_data):
    nmds_result = nuee.metaMDS(species_data, k=2, distance="bray", trace=True)
    return


@app.cell
def _(nuee, species_data):
    shannon_div = nuee.shannon(species_data)
    simpson_div = nuee.simpson(species_data)
    richness = nuee.specnumber(species_data)
    fisher_alpha = nuee.fisher_alpha(species_data)
    return


@app.cell
def _(env_data, nuee, species_data):
    rda_result = nuee.rda(species_data, env_data, scale=True)
    explained_var = rda_result.explained_variance_ratio[:3]
    return (rda_result,)


@app.cell
def _(nuee, rda_result):
    nuee.biplot(rda_result)
    return


@app.cell
def _(nuee, species_data):
    bray_dist = nuee.vegdist(species_data, method="bray")
    jaccard_dist = nuee.vegdist(species_data, method="jaccard")
    euclidean_dist = nuee.vegdist(species_data, method="euclidean")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
