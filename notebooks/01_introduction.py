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
        # Chapter 1: Data Science with Python

        ## Learning Objectives

        By the end of this chapter, you will:

        - Understand how data science relates to statistics
        - Be ready to work in a Python programming environment
        - Know how to perform basic operations in Python
        - Understand the main data types in Python
        - Know how to install and load packages

        ---

        A data science project comprises three main stages. First, you must **collect data** and compile it appropriately. This may involve downloading existing datasets, running an experimental design, or conducting a survey (observational study). Compiling data into an importable format is often a long and tedious task. Next, you **investigate the collected data**—that is, you visualize it, apply models, and test hypotheses. Finally, **communicating results** consists of presenting the knowledge emerging from your analysis in visual and narrative form, *using language adapted to your audience*, whether they are experts or novices, journal reviewers or managers. Following [Grolemund and Wickham (2018)](http://r4ds.had.co.nz/introduction.html), the analysis structure follows this workflow:

        ```
        Import → Tidy → Transform → Visualize → Model → Communicate
                         ↑_________|        |_______|
                              Understand
        ```

        The overarching framework is **Programming**. Yes, you will need to write code. But as indicated in the opening, this is not a programming course, and we will favor intuitive approaches.

        ## Statistics or Data Science?

        According to [Whitlock and Schluter (2015)](http://whitlockschluter.zoology.ubc.ca/), statistics is the *study of methods for describing and measuring aspects of nature from samples*. For [Grolemund and Wickham (2018)](http://r4ds.had.co.nz/introduction.html), data science is *an exciting discipline that allows you to turn raw data into understanding, insight, and knowledge*. Yes, *exciting*! The difference between the two fields of expertise is subtle, and some people see only a difference in tone.

        > "Data Science is statistics on a Mac." — Big Data Borat

        Confined to its traditional applications, statistics is more focused on defining experimental designs and performing hypothesis tests, while data science is less linear, particularly in its analysis phase, where new questions (and therefore new hypotheses) can be asked as the analysis progresses. This generally happens more when facing numerous observations with many measured parameters.

        The quantity of data and measurements we now have access to thanks to relatively inexpensive measurement and storage technologies makes data science a particularly attractive discipline, not to say [sexy](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century).

        ## Getting Started with Python

        [Python](https://www.python.org) is a programming language created by Guido van Rossum, initially released in 1991. Python is consistently ranked among [the most used programming languages in the world](https://www.tiobe.com/tiobe-index/). Python is a dynamic language, meaning code can be executed line by line or block by block: a major advantage for activities requiring frequent interactions. While Python is used for many purposes (web development, automation, AI), it has become a privileged tool in data science due to recent developments in analysis, modeling, and visualization packages, several of which will be used in this manual.

        Learning a programming language is a bit like learning a spoken language. At first, Python code may seem incomprehensible. And facing your keyboard, you're not quite sure how to express what you want. As you learn, symbols, functions, and style become increasingly familiar, and you gradually learn to translate what you want to accomplish into code. Just as a language is learned by speaking it in everyday life, a programming language is best learned by solving your own problems.

        ## Setting Up Your Workflow

        For this book, we recommend using **uv** for Python installation and package management. uv is a modern, fast Python package installer and resolver written in Rust.

        ### Installing Python with uv

        **Recommended installation**. First, install uv:

        **On macOS and Linux:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

        **On Windows:**
        ```powershell
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

        Then install Python:
        ```bash
        uv python install
        ```

        uv will automatically download and manage Python versions for you. It's much faster than traditional pip and handles dependency resolution more reliably.

        ### Installing Required Packages

        To install all packages needed for this book:

        ```bash
        uv pip install pandas numpy scipy statsmodels scikit-learn \
                       lets-plot nuee marimo pymc
        ```

        Or if you have a `requirements.txt` file:

        ```bash
        uv pip install -r requirements.txt
        ```

        ### Working with Marimo Notebooks

        This book uses [Marimo](https://marimo.io), a reactive notebook environment for Python. Marimo notebooks are:

        - **Reactive**: When you change a cell, dependent cells automatically update
        - **Reproducible**: No hidden state or execution order issues
        - **Git-friendly**: Notebooks are stored as pure Python scripts
        - **Interactive**: Built-in UI elements for exploration

        To run a notebook:
        ```bash
        marimo run 01_introduction.py
        ```

        To edit a notebook:
        ```bash
        marimo edit 01_introduction.py
        ```

        ## First Steps with Python

        Python doesn't work with menus or mouse clicking. You will need to enter commands with your keyboard, which you will learn by heart over time, or find by searching the internet. From personal experience, when I work with Python, I always have a browser open ready to receive a question.

        The following steps are first steps. They won't make you a programming black belt. Most Python users learned by practicing on their own data, running into obstacles, learning how to overcome or bypass them...

        For now, let's see if Python is as free as claimed.

        > "Freedom is the freedom to say that two plus two makes four. If that is granted, all else follows." — George Orwell, 1984
        """
    )
    return


@app.cell
def _(mo):
    mo.md("Let's verify that basic arithmetic works:")
    return


@app.cell
def __():
    # Basic arithmetic
    result = 2 + 2
    print(f"2 + 2 = {result}")
    return result,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Basic Arithmetic Operations

        Mathematical operations work as you would expect:
        """
    )
    return


@app.cell
def __():
    print(f"67.1 - 43.3 = {67.1 - 43.3}")
    print(f"2 * 4 = {2 * 4}")
    print(f"1 / 2 = {1 / 2}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Exponents can be written using `**` (Python standard) or `^` via numpy:
        """
    )
    return


@app.cell
def __():
    print(f"2**4 = {2**4}")

    # Use spaces around operators (except **) to make code clearer
    result_div = 1 / 2
    print(f"1 / 2 = {result_div}")
    return result_div,


@app.cell
def _(mo):
    mo.md(
        r"""
        Python doesn't read anything after the `#` symbol. This allows you to comment your code. Note also that the last operation includes spaces around operators. In this case (not always), spaces don't matter: they just help clarify the code. There are style guides for writing Python code. I **strongly** recommend **meticulously** following the [PEP 8](https://pep8.org/) style guide or using automated formatters like `black` or `ruff`.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Variable Assignment

        Assigning objects to variables is fundamental in programming. In Python, we assign using the equals sign `=`:
        """
    )
    return


@app.cell
def __():
    a = 3
    print(f"a = {a}")
    return a,


@app.cell
def _(mo):
    mo.md(
        r"""
        Technically, `a` points to the integer 3. Consequently, we can perform operations on `a`:
        """
    )
    return


@app.cell
def _(a):
    print(f"a * 6 = {a * 6}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        If you try to use `A` (uppercase), you'll get an error because Python is case-sensitive. Using the wrong case leads to errors.

        **Note**: Error messages are not always clear, but you will learn to understand them. In any case, they are meant to help you. Read them carefully!

        Variable names must always start with a letter and should not contain reserved characters (spaces, `+`, `*`). In variable definitions, use underscores `_` to separate words. Avoid using uppercase for naming objects (see [PEP 8](https://pep8.org/) for object naming conventions).

        **Note**: At this stage, you will probably be more comfortable copy-pasting these commands into your terminal.
        """
    )
    return


@app.cell
def __():
    # Example with meaningful variable names
    yield_per_tree = 50  # apples/tree
    number_of_trees = 300  # trees
    total_apples = yield_per_tree * number_of_trees
    print(f"Total apples: {total_apples}")
    return number_of_trees, total_apples, yield_per_tree


@app.cell
def _(mo):
    mo.md(
        r"""
        Like most programming languages, Python respects conventions for [order of operations](https://en.wikipedia.org/wiki/Order_of_operations):
        """
    )
    return


@app.cell
def __():
    result_ops = 10 - 9**0.5 * 2
    print(f"10 - 9^0.5 * 2 = {result_ops}")
    return result_ops,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## About This Book

        ### Target Audience

        This book is designed for:
        - Graduate students and researchers in ecology and environmental science
        - Agronomists and agricultural systems researchers
        - Conservation biologists
        - Environmental engineers

        ### Prerequisites

        - Basic knowledge of ecology and statistics
        - Some programming experience (helpful but not required)
        - Familiarity with linear algebra concepts

        ### Book Structure

        The book progresses from basic Python to advanced ecological modeling through 13 chapters:

        **Part I: Foundations**
        1. Introduction to Data Science with Python
        2. Python Fundamentals for Ecological Analysis
        3. Data Organization with Pandas
        4. Data Visualization with lets-plot
        5. Reproducible Science and Version Control

        **Part II: Statistical Analysis**
        6. Biostatistics
        7. Bayesian Biostatistics
        8. Ordination and Classification

        **Part III: Data Quality and Advanced Methods**
        9. Outlier Detection and Missing Data Imputation
        10. Machine Learning for Ecology
        11. Time Series Analysis
        12. Spatial Data Analysis
        13. Mechanistic Modeling

        ### Python Ecosystem for Ecology

        This book uses the following key packages:

        - **pandas**: Data manipulation and analysis
        - **numpy**: Numerical computing
        - **scipy**: Scientific computing and statistics
        - **statsmodels**: Statistical modeling
        - **scikit-learn**: Machine learning
        - **pymc**: Bayesian statistical modeling
        - **nuee**: Community ecology (Python port of R's vegan package)
        - **lets-plot**: Grammar of graphics for Python (similar to ggplot2)
        - **marimo**: Reactive notebook environment

        All packages are selected for compatibility and ease of use in ecological research.

        ## Learning to Code

        Like a spoken language, you only learn to express yourself in a programming language by putting yourself to the test, which you will do throughout this course. To encourage you, here are some tips for learning to code in Python:

        - **Python dislikes ambiguity**: A single misplaced comma and it doesn't know what to do. This can be frustrating at first, but this rigidity is necessary for scientific computing.
        - **Copy-paste is your friend**: Keeping in mind that you are responsible for your code and respecting copyrights, don't be afraid to copy-paste lines of code and customize them afterward.
        - **The error you get: others have gotten it before you**: The Q&A site [Stack Overflow](https://stackoverflow.com/questions/tagged/python) is an invaluable resource where people who posted questions received answers from experts (the best answers and questions appear first). Learn to search intelligently for answers by formulating your questions precisely!
        - **Study and practice**: Error messages in Python are common, even among experienced people. The best way to learn a language is to speak it, study its quirks, test them in conversation, etc.

        ### Next Steps

        In the next chapter, we'll dive deeper into Python fundamentals, covering:
        - Data types in detail (integers, floats, strings, booleans)
        - Collections (lists, tuples, dictionaries, sets)
        - Functions and how to create them
        - Loops and conditionals
        - Working with packages

        Let's get started!
        """
    )
    return


if __name__ == "__main__":
    app.run()
