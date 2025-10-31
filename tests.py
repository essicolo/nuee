import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import nuee
    return (nuee,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # MetaMDS

    ```{R}
    library(vegan)
    data(varespec)
    metaMDS(varespec, k=2, distance = "bray")
    ```
    """
    )
    return


@app.cell
def _(nuee):
    varespec = nuee.datasets.varespec()
    res = nuee.metaMDS(varespec, k=2, distance="bray")
    print(res.stress)                 # final stress
    print(res.best_run, res.best_stress)
    print(res.stress_history[:3])     # first few runs w/ stress + iter count
    print(res.transformations)        # ['sqrt', 'wisconsin']
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Diversity

    METTRE À JOUR MON BCI!!!!!!!!!!!!!!!


    ```
    > library(vegan)
    + data(BCI)
    + mean(diversity(BCI, index="shannon"))
    + mean(diversity(BCI, index="simpson"))
    + mean(specnumber(BCI))
    [1] 3.82084
    [1] 0.9590064
    [1] 90.78
    ```
    """
    )
    return


@app.cell
def _(nuee):
    species = nuee.datasets.BCI()
    shannon_div = nuee.shannon(species)
    simpson_div = nuee.simpson(species)
    richness = nuee.specnumber(species)

    print(f"Mean Shannon diversity: {shannon_div.mean()}")
    print(f"Mean Simpson diversity: {simpson_div.mean()}")
    print(f"Mean species richness: {richness.mean()}")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
