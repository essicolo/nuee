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
        # Chapter 2: Python Fundamentals for Ecological Analysis

        ## Learning Objectives

        By the end of this chapter, you will be able to:

        - Distinguish the main data types in Python
        - Work with collections (lists, tuples, dictionaries, sets)
        - Create and use functions
        - Use loops and conditionals
        - Install and import packages

        ---

        ## Data Types

        So far, we have mainly used **integers** (`int`) and **floating-point numbers** (`float`). Python includes other important types. A **string** (`str`) contains one or more characters. It is defined with double quotes `" "` or single quotes `' '`. There is no strict standard on which to use, but generally, we use single quotes for short expressions containing a single word or sequence of letters, and double quotes for sentences. One reason: double quotes are useful for including single quotes in a string.
        """
    )
    return


@app.cell
def __():
    # String examples
    import numpy as np
    import pandas as pd

    a = "The bear"
    b = "polar"
    combined = f"{a} {b}"
    print(combined)

    # Count characters
    print(f"Number of characters: {len(combined)}")
    return a, b, combined, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Boolean Data

        In scientific computing, it's common to make queries about whether a result is true or false:
        """
    )
    return


@app.cell
def __():
    a_num = 17
    print(f"a_num < 10: {a_num < 10}")
    print(f"a_num > 10: {a_num > 10}")
    print(f"a_num == 10: {a_num == 10}")
    print(f"a_num != 10: {a_num != 10}")
    print(f"a_num == 17: {a_num == 17}")
    print(f"not (a_num == 17): {not (a_num == 17)}")
    return a_num,


@app.cell
def _(mo):
    mo.md(
        r"""
        I just introduced a new data type: **boolean** (`bool`), which can only take two states - `True` or `False`. Note that the symbol `=` is reserved for assigning objects: for equality tests, we use the double equals `==`, or `!=` for inequality. To invert a boolean value, we use the keyword `not`.

        ## Collections

        The previous exercises introduced the most important default data types in Python for scientific computing: `int` (integer), `float` (real number), `str` (string), and `bool` (boolean). Others will be added throughout the course, such as categories and datetime units.

        When performing computational operations in science, we rarely use single values. We prefer to organize and process them as collections. Python offers several important collection types: **lists**, **tuples**, **dictionaries**, **sets**, and (via NumPy) **arrays**.

        ### Lists

        **Lists** are ordered sequences of items that can be of different types. A list is delimited by square brackets `[]` with elements separated by commas.
        """
    )
    return


@app.cell
def __():
    # List example
    species = ["Petromyzon marinus", "Lepisosteus osseus", "Amia calva", "Hiodon tergisus"]
    print(species)
    return species,


@app.cell
def _(mo):
    mo.md(
        r"""
        To access list elements, we use square brackets with the position. **Important**: Python uses **0-based indexing**, meaning the first element is at position 0, not 1!
        """
    )
    return


@app.cell
def _(species):
    print(f"First species: {species[0]}")
    print(f"Second species: {species[1]}")
    print(f"First three: {species[0:3]}")  # Slicing: start at 0, stop before 3
    print(f"Selected species: {[species[0], species[2]]}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The notation `0:3` creates a slice from position 0 up to (but not including) position 3. You can use negative indices to count from the end, and you can modify lists:
        """
    )
    return


@app.cell
def __():
    species_mod = ["Petromyzon marinus", "Lepisosteus osseus", "Amia calva", "Hiodon tergisus"]

    # Add an element
    species_mod.append("Cyprinus carpio")
    print(f"After append: {species_mod}")

    # Change an element
    species_mod[2] = "Lepomis gibbosus"
    print(f"After modification: {species_mod}")
    return species_mod,


@app.cell
def _(mo):
    mo.md(
        r"""
        ### NumPy Arrays

        While lists are flexible, **NumPy arrays** are preferred for numerical work because they:
        - Are much faster for numerical operations
        - Support vectorized operations (operations on entire arrays at once)
        - Use less memory
        - Are the foundation for pandas, scipy, and other scientific packages
        """
    )
    return


@app.cell
def _(np):
    # Create arrays
    abundance = np.array([12, 8, 15, 3, 22, 7, 18, 5])
    biomass = np.array([1.2, 0.8, 2.1, 0.3, 3.4, 0.9, 2.8, 0.6])

    # Vectorized operations
    density = abundance / biomass  # Element-wise division
    log_abundance = np.log(abundance + 1)  # Log transformation

    # Basic statistics
    print("Abundance Statistics:")
    print(f"  Mean: {np.mean(abundance):.2f}")
    print(f"  Median: {np.median(abundance):.2f}")
    print(f"  Std Dev: {np.std(abundance):.2f}")
    print(f"  Range: {np.min(abundance)} - {np.max(abundance)}")
    return abundance, biomass, density, log_abundance


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Matrices (2D Arrays)

        A **matrix** is a 2-dimensional array. In ecology, we rarely go beyond two dimensions, although N-dimensional arrays are common in mathematical modeling. In Python (via NumPy), we can assign row and column names using pandas DataFrames, but NumPy matrices are also useful:
        """
    )
    return


@app.cell
def _(np):
    # Create a matrix
    mat = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    print("Matrix:")
    print(mat)
    print(f"\nShape: {mat.shape}")  # (rows, columns)
    print(f"First row: {mat[0, :]}")
    print(f"First column: {mat[:, 0]}")
    return mat,


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Dictionaries

        **Dictionaries** are unordered collections of key-value pairs. They are heterogeneous and can contain any type of object, even other dictionaries. Each element can be identified by a key.
        """
    )
    return


@app.cell
def __():
    # Dictionary example
    my_dict = {
        "species": ["Petromyzon marinus", "Lepisosteus osseus", "Amia calva", "Hiodon tergisus"],
        "site": "A101",
        "weather_stations": ["746583", "783786", "856363"]
    }

    print(f"Species: {my_dict['species']}")
    print(f"Site: {my_dict['site']}")
    return my_dict,


@app.cell
def _(mo):
    mo.md(
        r"""
        **Exercise**: Access the second element of the species list in `my_dict`.

        ### DataFrames

        The most important collection type for data science is the **DataFrame** (from pandas). Technically, it's a table where each column can have a different data type. DataFrames are similar to R's `data.frame` or spreadsheet tables.
        """
    )
    return


@app.cell
def _(pd):
    # Create a DataFrame
    df = pd.DataFrame({
        "species": ["Petromyzon marinus", "Lepisosteus osseus", "Amia calva", "Hiodon tergisus"],
        "weight": [10, 13, 21, 4],
        "length": [35, 44, 50, 8]
    })

    print(df)
    print(f"\nData types:\n{df.dtypes}")
    return df,


@app.cell
def _(mo):
    mo.md(
        r"""
        DataFrame columns can be accessed in several ways:
        """
    )
    return


@app.cell
def _(df):
    # Access columns
    print("Weight column:")
    print(df["weight"])
    print("\nMultiple columns:")
    print(df[["weight", "length"]])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Functions

        When you write a command followed by parentheses, like `len(...)`, you're calling a function. Very generally, a function transforms something into something else.

        For example, the `mean()` function takes a collection of numbers as input and returns... you guessed it.
        """
    )
    return


@app.cell
def _(df, np):
    mean_weight = np.mean(df["weight"])
    print(f"Mean weight: {mean_weight}")
    return mean_weight,


@app.cell
def _(mo):
    mo.md(
        r"""
        You can create your own functions. Let's say you want to create a function that calculates the output of $x^3 - 2y + a$. To get the answer, you need arguments `x`, `y`, and `a`. The function operation is straightforward:
        """
    )
    return


@app.cell
def __():
    def operation_f(x, y, a=10):
        """
        Calculate x^3 - 2*y + a

        Parameters:
        -----------
        x : float
            Base value
        y : float
            Multiplier value
        a : float, optional
            Additive constant (default=10)

        Returns:
        --------
        float
            Result of the calculation
        """
        return x**3 - 2*y + a

    # Test the function
    result = operation_f(x=2, y=3, a=1)
    print(f"operation_f(2, 3, 1) = {result}")
    return operation_f, result


@app.cell
def _(mo):
    mo.md(
        r"""
        Note that `a` has a default value. The function output is specified by `return`. You can now use the `operation_f` function as needed.

        Such a function may seem trivial, but using custom functions allows you to avoid repeating the same operation multiple times in a workflow, reducing code duplication and potential errors. Personally, I use functions mainly to generate custom plots.

        **Exercise**: To gain autonomy, you'll need to be able to find the names of commands you need to perform desired tasks. This can be frustrating, but you'll feel increasingly comfortable with Python day by day. The exercise here is to find on your own the command that allows you to measure the length of a list.

        ## Loops

        Loops allow you to perform the same series of operations on multiple objects. To continue our example, we want to get the result of operation_f for parameters stored in this DataFrame:
        """
    )
    return


@app.cell
def _(pd):
    params = pd.DataFrame({
        "x": [2, 4, 1, 5, 6],
        "y": [3, 4, 8, 1, 0],
        "a": [6, 1, 8, 2, 5]
    })
    print(params)
    return params,


@app.cell
def _(mo):
    mo.md(
        r"""
        We can use a loop to calculate results for each row:
        """
    )
    return


@app.cell
def _(operation_f, params):
    # Using a for loop
    operation_res = []
    for i in range(len(params)):
        res = operation_f(
            x=params.loc[i, "x"],
            y=params.loc[i, "y"],
            a=params.loc[i, "a"]
        )
        operation_res.append(res)

    print(f"Results: {operation_res}")
    return i, operation_res, res


@app.cell
def _(mo):
    mo.md(
        r"""
        We can add these results to our DataFrame:
        """
    )
    return


@app.cell
def _(operation_res, params):
    params_with_results = params.copy()
    params_with_results["results"] = operation_res
    print(params_with_results)
    return params_with_results,


@app.cell
def _(mo):
    mo.md(
        r"""
        **Pythonic alternative**: In Python, we often use list comprehensions or `apply()` methods for more concise code:
        """
    )
    return


@app.cell
def _(operation_f, params):
    # List comprehension (more Pythonic)
    results_lc = [
        operation_f(row.x, row.y, row.a)
        for _, row in params.iterrows()
    ]

    # Or using apply (even more Pythonic for DataFrames)
    results_apply = params.apply(
        lambda row: operation_f(row.x, row.y, row.a),
        axis=1
    )

    print(f"List comprehension: {results_lc}")
    print(f"Apply method: {list(results_apply)}")
    return results_apply, results_lc


@app.cell
def _(mo):
    mo.md(
        r"""
        ### While Loops

        **`while` loops** perform an operation as long as a criterion is not met. They are useful for operations seeking convergence. They are rarely used in typical workflows, so I'll cover them briefly:
        """
    )
    return


@app.cell
def _(np):
    x = 100
    iterations = []

    while x > 1.1:
        x = np.sqrt(x)
        iterations.append(x)

    print("Iteration values:")
    for val in iterations:
        print(f"  {val:.4f}")
    return iterations, val, x


@app.cell
def _(mo):
    mo.md(
        r"""
        We initialized `x` to a value of 100. Then, while the test `x > 1.1` is true, we assign to `x` the new value calculated by extracting the square root of the previous value of `x`.

        ## Conditionals: `if`, `elif`, `else`

        > If condition 1 is met, execute instruction suite 1. If condition 1 is not met, and if condition 2 is met, execute instruction suite 2. Otherwise, execute instruction suite 3.

        This is how we express a sequence of conditions. Let's take a simple example of discretizing a continuous value. If $x < 10$, it is classified as low. If $10 \leq x < 20$, it is classified as medium. If $x \geq 20$, it is classified as high. Let's put this classification in a function:
        """
    )
    return


@app.cell
def __():
    def classification(x, lim1=10, lim2=20):
        """
        Classify a value into low, medium, or high categories.

        Parameters:
        -----------
        x : float
            Value to classify
        lim1 : float
            Lower threshold (default=10)
        lim2 : float
            Upper threshold (default=20)

        Returns:
        --------
        str
            Classification category
        """
        if x < lim1:
            category = "low"
        elif x < lim2:
            category = "medium"
        else:
            category = "high"
        return category

    # Test the function
    print(f"classification(-10) = {classification(-10)}")
    print(f"classification(15.4) = {classification(15.4)}")
    print(f"classification(1000) = {classification(1000)}")
    return classification,


@app.cell
def _(mo):
    mo.md(
        r"""
        A condition is defined with `if`, followed by the test (true or false). If the test returns `True`, the instruction in the indented block is executed. If it's false, we move to the next one.

        **Exercise**: Explore the pandas `cut()` function and numpy's `where()` function, and think about how they could be used to perform discretization more efficiently than with `if` and `else`.

        ## Installing and Loading Packages

        Most general operations (like square roots, statistical tests, matrix and table management, graphs, etc.) are accessible through Python's built-in modules. However, teams have developed many packages to meet their specialized needs and made them available to the public through repositories like PyPI (Python's App Store), conda repositories, GitHub, etc.

        The command to install a package is:
        ```bash
        uv pip install package-name
        # or
        pip install package-name
        ```

        For example, to install `pandas`:
        ```bash
        uv pip install pandas
        ```

        ### Importing Packages

        Packages are like specialized applications you install on a mobile phone. To use them, you must import them. Generally, I import all necessary packages at the very beginning of my notebook:
        """
    )
    return


@app.cell
def __():
    # Standard imports for data science
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from scipy import optimize
    import matplotlib.pyplot as plt

    # For this book, we'll also use:
    # import nuee  # Community ecology
    # from lets_plot import *  # Visualization

    print("Packages imported successfully!")
    return optimize, plt, stats


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Common Import Conventions

        The Python community has established standard import conventions:

        ```python
        import numpy as np                    # Numerical computing
        import pandas as pd                   # Data analysis
        import matplotlib.pyplot as plt       # Plotting
        import scipy.stats as stats          # Statistics
        from lets_plot import *              # ggplot2-like plotting
        import nuee                          # Community ecology
        ```

        These conventions make code more readable and consistent across the community.

        ## Summary

        In this chapter, you learned:

        - **Data types**: integers, floats, strings, booleans
        - **Collections**: lists, tuples, dictionaries, sets, NumPy arrays, pandas DataFrames
        - **Functions**: Creating and using custom functions
        - **Loops**: `for` loops, `while` loops, and Pythonic alternatives (list comprehensions, `apply`)
        - **Conditionals**: `if`, `elif`, `else` statements
        - **Packages**: Installing with `uv pip install` and importing with `import`

        ## Tips for Learning Python

        - **Python dislikes ambiguity**: A simple misplaced comma and it doesn't know what to do. This can be frustrating at first, but this rigidity is necessary for scientific computing.
        - **Copy-paste is your friend**: While being responsible for your code and respecting copyrights, don't be afraid to copy-paste code and customize it.
        - **Your error has been encountered before**: The Q&A site [Stack Overflow](https://stackoverflow.com/questions/tagged/python) is an invaluable resource where people's questions have been answered by experts (best answers and questions appear first). Learn to search intelligently by precisely formulating your questions!
        - **Study and practice**: Error messages are common in Python, even among experienced users. The best way to learn a language is to speak it, study its quirks, test them in practice, etc.

        ## Exercise

        To familiarize yourself with Python and this notebook environment:

        1. Try modifying some of the code cells above
        2. Create your own variables and perform operations on them
        3. Write a simple function that takes two numbers and returns their sum
        4. Create a list of numbers and calculate their mean using `np.mean()`
        5. Create a simple DataFrame with ecological data (e.g., species names and counts)

        In the next chapter, we'll dive deep into data organization with pandas, learning how to manipulate, transform, and clean ecological datasets efficiently.
        """
    )
    return


if __name__ == "__main__":
    app.run()
