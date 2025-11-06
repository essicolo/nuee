import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Chapter 4: Data Visualization

        ## Learning Objectives

        By the end of this chapter, you will:

        - Understand the importance of data exploration through visualization
        - Understand general guidelines for creating appropriate graphics
        - Understand the difference between imperative and declarative approaches
        - Be able to create scatter plots, lines, histograms, bar charts, and boxplots
        - Know how to export graphics for publication

        ---

        Creating graphics is a common task in scientific workflows. A well-designed graphic is dense with information. Data visualization allows us to explore tables and create visual elements for publication, conveying information that would otherwise be difficult or impossible to communicate adequately.

        Most graphics you generate won't be destined for publication. They'll probably first aim to explore data, allowing you to highlight new perspectives.

        ## Why Visualize Graphically?

        Let's take two variables, $X$ and $Y$. You calculate their mean, standard deviation, and correlation. Can summary statistics tell you everything about your data? Not quite.

        To demonstrate that these statistics won't teach you much about data structure, [Matejka and Fitzmaurice (2017)](https://www.autodeskresearch.com/publications/samestats) generated 12 datasets of $X$ and $Y$, each having practically the same statistics. But with very different structures! This is the famous **Datasaurus Dozen** - a powerful reminder that you must always visualize your data.
        """
    )
    return


@app.cell
def __():
    # Essential imports
    import pandas as pd
    import numpy as np
    from lets_plot import *

    # Initialize lets-plot for notebook use
    LetsPlot.setup_html()

    # For classic datasets
    from sklearn.datasets import load_iris

    print("lets-plot initialized successfully!")
    return LetsPlot, load_iris, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Five Qualities of Good Visualization

        Alberto Cairo, researcher specialized in data visualization, published in 2016 [*The Truthful Art*](http://www.thefunctionalart.com/p/the-truthful-art-book.html), noting five qualities of well-designed visualization:

        > **1. It is truthful**, as it is based on thorough and honest research.

        **Present data according to the most accurate interpretation**. Avoid cherry-picking (removing inconvenient data) and over-interpreting (seeing patterns in noise).

        > **2. It is functional**, constituting an accurate representation of data, built to let observers make consequential initiatives.

        Choosing the right graphic is a **rational approach to the objective** of presenting information, not just aesthetics.

        > **3. It is attractive** and intriguing, aesthetically pleasing for the target audience.

        **Present information, not decorations**. Minimize unnecessary elements. Provide maximum information with minimum graphical elements.

        > **4. It is insightful**, revealing scientific evidence otherwise difficult to access.

        **Generate an idea at a glance**. A good visualization elicits a "eureka" moment.

        > **5. It is enlightening**, changing our perception for the better when we accept its evidence.

        **Select the information to transmit** carefully and purposefully.

        ## Choosing the Right Plot Type

        The site [from data to viz](https://www.data-to-viz.com/) provides decision trees guiding you toward appropriate options for presenting your data. [Ann K. Emery's site](https://annkemery.com/essentials/) presents guidelines for the right graphic according to data in hand.

        Key considerations:

        1. **Think about your message**: Compare categories, visualize change over time, show relationships, or display spatial distribution?
        2. **Try different representations**: You might need more than one graphic
        3. **Organize your data**: Follow tidy data principles (Chapter 3)
        4. **Test the result**: "Hey, what do you understand from this?" If they shrug, reevaluate

        ## Grammar of Graphics with lets-plot

        **lets-plot** is Python's implementation of the Grammar of Graphics, inspired by R's ggplot2. The approach is **declarative** - you specify *what* to display, not *how* to display it.

        > Declarative visualization lets you think about data and relationships, rather than accessory details.

        A grammar of graphics has 5 components:

        1. **Data**: Your DataFrame
        2. **Geometries** (markers): Points, lines, polygons, bars, etc. (`geom_point()`, `geom_line()`)
        3. **Aesthetics** (encoded attributes): Position, size, color, shape
        4. **Global attributes**: Constant attributes (don't depend on a variable)
        5. **Themes**: Customize how the graphic is rendered

        Workflow:
        ```
        With my DataFrame,
        Create a marker (
            encode(position X = column A,
                   position Y = column B,
                   color = column C),
            global shape = 1)
        With a black and white theme
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## First lets-plot Example

        Let's load the classic iris dataset (published in 1936 by biostatistician Ronald Fisher):
        """
    )
    return


@app.cell
def _(load_iris, pd):
    # Load iris dataset
    iris_data = load_iris(as_frame=True)
    iris = iris_data.frame
    iris['species'] = iris_data.target_names[iris_data.target]

    print("Iris dataset:")
    print(iris.head())
    print(f"\nColumns: {iris.columns.tolist()}")
    return iris, iris_data


@app.cell
def _(aes, geom_point, ggplot, iris):
    # Basic scatter plot
    (ggplot(iris, aes(x='sepal length (cm)', y='sepal width (cm)'))
     + geom_point())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Following the lets-plot grammar:

        1. `ggplot(data, aes(...))` - specify data and aesthetic mappings
        2. Add geometries with `+` (just like ggplot2 in R)
        3. `geom_point()` creates points

        ### Adding Color, Size, and Transparency

        We can encode additional variables:
        """
    )
    return


@app.cell
def _(aes, geom_point, ggplot, iris):
    # Scatter plot with color and size
    (ggplot(iris, aes(x='sepal length (cm)',
                      y='sepal width (cm)',
                      color='species',
                      size='petal length (cm)'))
     + geom_point(alpha=0.6))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Available Geometries

        lets-plot provides many geometry types:

        - `geom_point()` - scatter plots
        - `geom_line()` - line plots
        - `geom_bar()`, `geom_col()` - bar charts
        - `geom_histogram()` - histograms
        - `geom_boxplot()` - box plots
        - `geom_density()` - density plots
        - `geom_smooth()` - smoothed trend lines
        - `geom_errorbar()`, `geom_pointrange()` - error bars
        - `geom_tile()` - heatmaps
        - etc.

        ## Available Aesthetics

        - Position: `x`, `y`, `z`
        - Size: `size`
        - Shape: `shape`
        - Color: `color` (outline), `fill` (interior)
        - Line type: `linetype`
        - Transparency: `alpha`

        ## Facets

        **Facets** split your plot into multiple subplots based on categorical variables:

        - `facet_wrap()` - wraps facets into rectangular layout
        - `facet_grid()` - creates a grid

        Facets are conventionally placed right after `ggplot()`:
        """
    )
    return


@app.cell
def _(aes, facet_wrap, geom_point, ggplot, iris):
    # Faceted plot
    (ggplot(iris, aes(x='sepal length (cm)', y='sepal width (cm)'))
     + facet_wrap('species', ncol=2)
     + geom_point(aes(color='species'), alpha=0.6))
    return


@app.cell
def _(aes, facet_grid, geom_point, ggplot, iris, pd):
    # Facet grid with discretized variable
    iris_disc = iris.copy()
    iris_disc['petal_cat'] = pd.cut(iris['petal length (cm)'],
                                     bins=[0, 2, 4, 6, 8],
                                     labels=['0-2', '2-4', '4-6', '6-8'])

    (ggplot(iris_disc, aes(x='sepal length (cm)', y='sepal width (cm)'))
     + facet_grid('species ~ petal_cat')
     + geom_point(aes(color='species'), alpha=0.5))
    return iris_disc,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Line Plots

        Lines show trends over time or ordered sequences. Let's create time series data:
        """
    )
    return


@app.cell
def _(np, pd):
    # Create time series
    np.random.seed(42)
    time_data = pd.DataFrame({
        'time': list(range(100)) * 3,
        'value': np.concatenate([
            np.cumsum(np.random.randn(100)) + 10,
            np.cumsum(np.random.randn(100)) + 15,
            np.cumsum(np.random.randn(100)) + 5
        ]),
        'tree': ['Tree 1'] * 100 + ['Tree 2'] * 100 + ['Tree 3'] * 100
    })

    print("Time series data (first 10 rows):")
    print(time_data.head(10))
    return time_data,


@app.cell
def _(aes, geom_line, ggplot, time_data):
    # Line plot
    (ggplot(time_data, aes(x='time', y='value', color='tree'))
     + geom_line())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Histograms

        Histograms show distribution of a continuous variable by dividing it into bins. The `bins` argument (like R's `breaks`) specifies the number of bins:
        """
    )
    return


@app.cell
def _(aes, geom_histogram, ggplot, iris):
    # Basic histogram
    (ggplot(iris, aes(x='petal length (cm)'))
     + geom_histogram(bins=30, fill='steelblue', color='white'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Histograms with Facets

        Facets help compare distributions across groups. Using `fill` colors the interior of bars:
        """
    )
    return


@app.cell
def _(aes, facet_grid, geom_histogram, ggplot, iris):
    # Faceted histogram
    (ggplot(iris, aes(x='petal length (cm)', fill='species'))
     + facet_grid('species ~ .')
     + geom_histogram(bins=20, color='white'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Note**: Bin width is critical! The same data with different bin widths can reveal different patterns. Experiment with the `bins` parameter during exploration.

        ## Boxplots

        Boxplots visualize distributions through quartiles. The box extends from Q1 (25th percentile) to Q3 (75th percentile), with a line at the median (Q2). Whiskers extend to 1.5 × IQR (interquartile range), and points beyond are outliers.
        """
    )
    return


@app.cell
def _(aes, geom_boxplot, ggplot, iris):
    # Basic boxplot
    (ggplot(iris, aes(x='species', y='petal length (cm)'))
     + geom_boxplot(fill='lightblue'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Boxplots with Data Points

        It's often helpful to show actual measurements alongside boxplots using jittered points:
        """
    )
    return


@app.cell
def _(aes, geom_boxplot, geom_jitter, ggplot, iris):
    # Boxplot with jittered points
    (ggplot(iris, aes(x='species', y='petal length (cm)'))
     + geom_boxplot(fill='lightblue', alpha=0.5)
     + geom_jitter(width=0.2, height=0, alpha=0.3, color='darkblue'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Bar Charts

        Bar charts compare values across categories. Let's create categorical data:
        """
    )
    return


@app.cell
def _(np, pd):
    # Create categorical data
    np.random.seed(42)
    bar_data = pd.DataFrame({
        'peatland': ['Beaumont', 'Mont-Blanc', 'Water Plant', 'North Bog', 'South Fen'],
        'yield_g': [145, 178, 156, 134, 167],
        'treatment': ['Control', 'Fertilized', 'Fertilized', 'Control', 'Control']
    })

    print("Bar chart data:")
    print(bar_data)
    return bar_data,


@app.cell
def _(aes, bar_data, geom_bar, ggplot):
    # Bar chart using geom_bar with stat='identity'
    (ggplot(bar_data, aes(x='peatland', y='yield_g'))
     + geom_bar(stat='identity', fill='coral'))
    return


@app.cell
def _(aes, bar_data, geom_bar, ggplot, position_dodge):
    # Grouped bar chart
    (ggplot(bar_data, aes(x='peatland', y='yield_g', fill='treatment'))
     + geom_bar(stat='identity', position=position_dodge()))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Customizing Plots: Themes and Labels

        lets-plot offers several built-in themes:

        - `theme_gray()` - default (ggplot2 style with gray background)
        - `theme_bw()` - black and white
        - `theme_light()` - light theme
        - `theme_minimal()` - minimal theme
        - `theme_classic()` - classic (no grid)
        - `theme_void()` - completely empty

        You can customize labels with `labs()`, `xlab()`, `ylab()`, and `ggtitle()`:
        """
    )
    return


@app.cell
def _(aes, geom_point, ggplot, ggtitle, iris, theme_classic, xlab, ylab):
    # Customized plot
    (ggplot(iris, aes(x='sepal length (cm)',
                      y='sepal width (cm)',
                      color='species'))
     + geom_point(size=3, alpha=0.6)
     + ggtitle("Iris Dataset: Sepal Dimensions")
     + xlab("Sepal Length (cm)")
     + ylab("Sepal Width (cm)")
     + theme_classic())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Advanced Customization with theme()

        The `theme()` function allows precise adjustments. You can customize axis titles, text sizes, legend position, and more:

        ```python
        (ggplot(data, aes(...))
         + geom_point()
         + theme_bw()
         + theme(axis_title=element_text(size=14),
                 axis_text=element_text(size=12),
                 legend_position='bottom'))
        ```

        For mathematical expressions in labels:
        ```python
        labs(x="Temperature (°C)", y="Density (kg m⁻³)")
        ```

        ## Adding Smooth Trend Lines

        `geom_smooth()` adds smoothed trend lines to visualize patterns:
        """
    )
    return


@app.cell
def _(aes, geom_point, geom_smooth, ggplot, iris):
    # Plot with smooth line
    (ggplot(iris, aes(x='sepal length (cm)', y='petal length (cm)'))
     + geom_point(aes(color='species'), alpha=0.4)
     + geom_smooth(method='loess', color='black'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Multiple Data Sources

        You can specify different data sources for different geometries:
        """
    )
    return


@app.cell
def _(aes, geom_point, ggplot, iris, pd):
    # Create additional data
    special_obs = pd.DataFrame({
        'sepal length (cm)': [5.0, 6.5, 7.5],
        'sepal width (cm)': [3.5, 3.0, 3.8]
    })

    # Plot with multiple data sources
    (ggplot(iris, aes(x='sepal length (cm)', y='sepal width (cm)'))
     + geom_point(aes(color='species'), alpha=0.3)
     + geom_point(data=special_obs, size=6, color='red', shape=4))
    return special_obs,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ecological Data Example: Cloudberry Study

        Let's create a comprehensive example using ecological data (cloudberry cultivation):
        """
    )
    return


@app.cell
def _(np, pd):
    # Create cloudberry dataset
    np.random.seed(42)

    cloudberry = pd.DataFrame({
        'peatland': ['BEAU'] * 10 + ['MB'] * 10 + ['WTP'] * 10,
        'longitude': np.random.normal(-71.2, 0.3, 30),
        'latitude': np.random.normal(48.5, 0.3, 30),
        'N_perc': np.random.uniform(0.5, 2.5, 30),
        'P_perc': np.random.uniform(0.05, 0.3, 30),
        'K_perc': np.random.uniform(0.1, 0.6, 30),
        'yield_g': np.random.normal(150, 50, 30).clip(0),
        'rings': np.random.poisson(8, 30)
    })

    print("Cloudberry dataset:")
    print(cloudberry.head())
    return cloudberry,


@app.cell
def _(aes, cloudberry, geom_point, ggplot):
    # Multi-aesthetic plot
    (ggplot(cloudberry, aes(x='N_perc', y='yield_g',
                            color='peatland', size='rings'))
     + geom_point(alpha=0.6))
    return


@app.cell
def _(aes, cloudberry, facet_grid, geom_boxplot, ggplot):
    # Faceted boxplot
    (ggplot(cloudberry, aes(x='peatland', y='yield_g', fill='peatland'))
     + geom_boxplot()
     + facet_grid('. ~ peatland', scales='free_x'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Saving Plots for Publication

        To export plots, use `ggsave()`:

        ```python
        # Save to PNG (raster format)
        p = ggplot(data, aes(...)) + geom_point()
        ggsave(p, 'figure1.png', dpi=300, width=8, height=6)

        # Save to PDF (vector format)
        ggsave(p, 'figure1.pdf', width=8, height=6)

        # Save to SVG (vector format, editable in Inkscape)
        ggsave(p, 'figure1.svg', width=8, height=6)
        ```

        **Resolution for publication**: 300 dpi or higher for raster formats.

        **Vector formats** (SVG, PDF): Scalable, editable in vector graphics software.

        **Raster formats** (PNG): Use for graphics with sharp color changes. PNG supports transparency.

        ## Summary

        In this chapter, you learned:

        - **Why visualize**: Statistics alone are insufficient (Datasaurus Dozen)
        - **Five qualities**: Truthful, functional, attractive, insightful, enlightening
        - **Grammar of graphics**: Declarative approach with data, geometries, aesthetics
        - **lets-plot basics**: `ggplot()` + geometries + aesthetics
        - **Plot types**: Scatter, line, histogram, boxplot, bar charts
        - **Facets**: Split plots by categorical variables
        - **Customization**: Themes, labels, colors, transparency
        - **Multiple data sources**: Different data for different layers
        - **Exporting**: Publication-quality graphics with `ggsave()`

        ### Key lets-plot Syntax Summary

        ```python
        from lets_plot import *
        LetsPlot.setup_html()

        # Basic structure
        (ggplot(data, aes(x='col1', y='col2'))
         + geom_point()
         + theme_bw())

        # With multiple aesthetics
        (ggplot(data, aes(x='x', y='y', color='group', size='value'))
         + geom_point(alpha=0.6)
         + facet_wrap('category'))

        # Saving
        ggsave(plot, "output.png", dpi=300, width=8, height=6)
        ```

        ### lets-plot to ggplot2 (R) Comparison

        | Component | lets-plot (Python) | ggplot2 (R) |
        |-----------|-------------------|-------------|
        | Initialize | `LetsPlot.setup_html()` | `library(ggplot2)` |
        | Basic plot | `ggplot(data, aes(...))` | `ggplot(data, aes(...))` |
        | Add layer | `+ geom_point()` | `+ geom_point()` |
        | Facets | `+ facet_wrap()` | `+ facet_wrap()` |
        | Themes | `+ theme_bw()` | `+ theme_bw()` |
        | Save | `ggsave(p, 'file.png')` | `ggsave('file.png')` |

        ### Tips for Effective Visualization

        1. **Start simple**: Basic plots first, add complexity as needed
        2. **Choose appropriately**: Match plot type to data and message
        3. **Use color purposefully**: Color is information
        4. **Label clearly**: Self-explanatory axes, titles, legends
        5. **Minimize clutter**: Remove unnecessary elements
        6. **Consider your audience**: Scientists vs. general public
        7. **Test it**: "What do you understand from this?"
        8. **Iterate**: Rarely perfect on first try

        ### Resources

        - [from data to viz](https://www.data-to-viz.com/) - Decision trees for plot selection
        - [ColorBrewer](http://colorbrewer2.org/) - Color palettes for maps and graphics
        - [lets-plot documentation](https://lets-plot.org/) - Complete API reference

        In the next chapter, we'll explore reproducible science and version control!
        """
    )
    return


if __name__ == "__main__":
    app.run()
