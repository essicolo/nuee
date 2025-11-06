import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Open Science and Reproducibility {#chapter-git}

        ***

        **Learning Objectives**:

        By the end of this chapter, you will be able to:

        - Explain the importance and challenges of open science
        - Organize your data (CSV format) and code (notebook format) to make your research reproducible
        - Create a repository on GitHub and manage its development

        ***
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pandas as pd
    return mo, np, pd


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## What is Open Science?

        Open science promotes the dissemination of knowledge through several key aspects:

        - **Open Methodology**: Scientific journals require detailed methodology descriptions to ensure data collection is well understood and can be replicated in future experiments. For example, the journal *Nature* created [Protocol Exchange](https://www.nature.com/protocolexchange/), "where the scientific community shares experimental expertise to accelerate research."

        - **Open Data**: By making our data public, we enable future researchers to improve knowledge, discover structures we may have missed, etc. In some cases, open data may be constrained by legal issues (private data) or ethical concerns (data that could be misused). In most cases, the benefits far outweigh the risks of data publication, and personal information can be removed. Journals like PLOS [require](https://blogs.plos.org/everyone/2014/02/24/plos-new-data-policy-public-access-data-2/) that minimal data needed to reproduce experiments be provided as supplementary material.

        - **Open Source Code**: Open source software, like Python, is free for most users. This allows anyone to use it, provided they have the hardware (a computer) and an internet connection. Similarly, Python code that generates results from your data can be made public under various permissive open source licenses (GPL, BSD, MIT, etc.). With data and code, your work can be reproduced.

        - **Open Peer Review**: Peer review is essential work in science. Traditionally, scientific publications are reviewed anonymously to avoid conflicts. Recently, journals like [Frontiers](https://www.frontiersin.org/about/review-system) have deployed open review modes, enabling (1) more constructive exchanges between authors and reviewers and (2) openly acknowledging reviewer contributions to the final article.

        - **Open Access**: Scientific publishers are [widely criticized](https://www.nature.com/articles/d41586-019-00492-4) for charging exorbitant fees to libraries and for individual article access, as well as excessive publication fees. In response, sites like [Sci-Hub](https://en.wikipedia.org/wiki/Sci-Hub) unlock millions of scientific articles for free. Additionally, reputable journals like PLOS and Frontiers publish articles directly on their websites so they can be freely downloaded.

        The lack of openness in science has led many scientists to speak of a reproducibility crisis ([Baker, 2016](https://www.nature.com/news/1-500-scientists-lift-the-lid-on-reproducibility-1.19970)). In this chapter, we'll explore techniques to make Python a tool that promotes open science. By the end of this chapter, you should be able to deploy your code to an online archive.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Reproducible Code

        The *British Ecological Society* offers guidelines for creating a reproducible workflow ([BES, 2017](https://www.britishecologicalsociety.org/wp-content/uploads/2017/12/guide-to-reproducible-code.pdf)). The following principles should be respected:

        - **Start your analysis from a copy of the raw data**. Data should be provided in an open format (CSV, JSON, SQLite, etc.). Avoid starting an analysis with a spreadsheet or proprietary software (that is not open source). In this sense, starting with Excel (`.xls` or `.xlsx`) should be avoided, as should data encoded for SPSS or SAS.

        - **All data operations should be performed with code, not manually**. Whether cleaning, merging, transforming, etc., these should be done with code. If it's a typo in a table, you can make an exception. But if you're removing outliers, for example, don't delete entries from your raw data. Similarly, don't transform your raw data outside of code. In short, your calculations should be able to run all at once, without intermediate manual operations.

        - **Separate your operations into logical thematic units**. For example, you could separate your code into parts: (i) load, merge, and clean data, (ii) analyze data, (iii) create outputs like tables and figures.

        - **Eliminate code duplication by creating custom functions**. Make sure to comment your functions in detail, explaining what is expected as inputs and outputs, what they do and why.

        - **Document your code and data within the notebooks or in a separate documentation file**.

        - **Any intermediate files should be separated from your raw data**.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Project Structure

        A computational project should be contained in a single folder. If you only have a few projects, it's easy to keep track. However, especially in a business environment, you might have to manage several projects simultaneously. Some organizations create project numbers: you'll benefit from naming your folders with these numbers, including a brief description. Personally, I organize my projects chronologically by year, with a description.

        ```
        📁 2019_bee-cranberry
        ```

        Note that I don't use spaces or special characters in the folder name to avoid potential errors with finicky software.

        Inside the project root folder, I include general information: source data (often Excel files), manuscript (thesis, article, etc.), specific documentation (for articles, I use Zotero, a reference manager), photos, and of course, my code folder (e.g., `python`).

        ```
        📁 2019_bee-cranberry
        ├─📁 documentation
        ├─📁 manuscript
        ├─📁 photos
        ├─📁 python
        └─📁 source
        ```

        If you write your manuscript within your code (in LaTeX, [Lyx](https://www.lyx.org/), markdown, or Marimo notebooks as we'll see later), you can include it in your computational folder.

        Inside the computational folder, you'll have your Python project and your sequenced notebooks. I use `01-`, not `1-`, to avoid having `10-` follow `1-` in alphanumeric sorting if I have more than 10 notebooks. I include a `README.md` file (`.md` extension for markdown), which contains general information about my computations. Raw data (`.csv`) are placed in a `data` folder, my graphics are exported to an `images` folder, my tables are exported to a `tables` folder, and my external functions are exported to a `lib` folder.

        ```
        📁 python
        ├─📁 data
        ├─📁 images
        ├─📁 lib
        ├─📁 tables
        ├─📄 pyproject.toml
        ├─📄 uv.lock
        ├─📄 01_clean_data.py
        ├─📄 02_data_mining.py
        ├─📄 03_data_analysis.py
        ├─📄 04_data_modeling.py
        └─📄 README.md
        ```

        I describe file names in the language useful for the final project deliverable, often in English for academic publications. I avoid uninformative file names, such as `01.py` or `plot1.png`, as well as capital letters, special characters, and spaces like in `Second try.py` (the `README.md` is an exception).

        To share a Python project folder, you just need to compress it (*zip*) and send it. For the code to work on another computer, links to data files to import or graphics to export must be relative to the Python file opened in your project, not the full path on your computer.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Relative vs. Absolute Paths

        When working with files in Python, always use relative paths from your project directory:
        """
    )
    return


@app.cell
def __(pd):
    # Good: relative path
    # data = pd.read_csv("data/bees.csv")

    # Bad: absolute path (won't work on another computer)
    # data = pd.read_csv("/Users/your_name/projects/2019_bee-cranberry/python/data/bees.csv")

    # Python's pathlib makes this even cleaner
    from pathlib import Path

    # project_root = Path.cwd()
    # data_path = project_root / "data" / "bees.csv"
    # data = pd.read_csv(data_path)
    return (Path,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Just like the BES, the nonprofit organization [rOpenSci](https://ropensci.org) offers [a guide on reproducibility](http://ropensci.github.io/reproducibility-guide/).
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Marimo Notebooks

        Reproducible code is well-documented code. The project structure presented above proposes segmenting code into multiple Python files. This approach is optional. If the computational file isn't too cumbersome, you can use just one, for example `stats.py`. Within Python scripts, you should comment your code to explain the steps:

        ```python
        #############
        ## Title 1 ##
        #############

        # Title 2
        ## Title 3
        data = pd.read_csv("data/bees.csv")  # specific comment
        ```

        A more user-friendly approach is to use **Marimo notebooks**. Marimo is a reactive notebook environment that stores notebooks as pure Python scripts, making them easy to version control and share. These course notes are entirely written in Marimo notebooks.

        ### Why Marimo?

        Marimo notebooks offer several advantages over traditional Jupyter notebooks:

        - **Reactive execution**: When you change a variable, all dependent cells automatically update
        - **Pure Python files**: Notebooks are stored as `.py` files, making them easy to version control with git
        - **No hidden state**: The execution order is always clear and reproducible
        - **Built-in version control**: Since they're Python files, they work seamlessly with git
        - **Easy to share**: Send a `.py` file, and others can run it directly

        ### Creating a Marimo Notebook

        To create a new Marimo notebook, use the command line:

        ```bash
        marimo edit my_analysis.py
        ```

        This will open a browser window with the Marimo editor. The notebook structure is simple:
        """
    )
    return


@app.cell
def __():
    # Example Marimo cell structure
    # Each cell is a Python function decorated with @app.cell

    # Cell 1: Import libraries
    # import numpy as np
    # import pandas as pd

    # Cell 2: Load data
    # data = pd.read_csv("data/cloudberry.csv")

    # Cell 3: Analyze data
    # summary = data.describe()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Markdown in Marimo

        Marimo uses `mo.md()` to display markdown-formatted text. You can use all standard markdown formatting:

        **Italic**: Surround text with single asterisks. For example, `*emphasized text*` becomes *emphasized text*.

        **Bold**: Surround text with double asterisks. For example, `**important text**` becomes **important text**.

        **Fixed width**: For inline code, surround text with backticks. For example, `` `variable_name` `` becomes `variable_name`.

        **Lists**: For numbered lists, use `1.`:

        ```
        1. First item
        1. Second item
        1. Third item
        ```

        becomes

        1. First item
        2. Second item
        3. Third item

        For bullet lists, use `-` or `*`:

        - Item one
        - Item two
        - Item three

        **Headers**: Titles are preceded by `#`. One `#` for level 1, two `##` for level 2, etc.

        ```
        # Main Title
        ## Section
        ### Subsection
        ```

        **Links**: Text in square brackets followed by URL in parentheses. For example, `[Python documentation](https://docs.python.org)` becomes [Python documentation](https://docs.python.org).

        **Equations**: Equations follow LaTeX syntax between single `$` for inline equations and double `$$` for display equations. For example, `$c = \sqrt{a^2 + b^2}$` becomes $c = \sqrt{a^2 + b^2}$.

        **Images**: To insert an image, `![image name](images/figure.png)`.

        A comprehensive list of markdown tags is available as a [cheat sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Running Marimo Notebooks

        To run a Marimo notebook:

        ```bash
        # Interactive mode (opens in browser)
        marimo edit notebook.py

        # Run mode (executes and shows output)
        marimo run notebook.py

        # Convert to HTML
        marimo export html notebook.py > output.html

        # Convert to markdown
        marimo export md notebook.py > output.md
        ```

        Marimo notebooks are reactive, meaning that when you change a cell, all cells that depend on it automatically re-execute. This ensures your notebook is always in a consistent state.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Introduction to GitHub

        The version control system **git** (open source) was created by Linus Torvalds, also known for creating Linux. **git** takes a snapshot of your project directory every time you *commit* a change. You can easily return to old versions if something goes wrong, and you can publish the final result on a hosting service that uses **git**.

        Several services exist to use **git** online, but GitHub is definitely the most widely used. The [GitHub](https://github.com/) platform has almost become a social network for development. GitHub, now owned by Microsoft, is not itself open source. If like me you have a preference for open source, I recommend the [GitLab](https://about.gitlab.com/) platform, which works much the same way as GitHub. In general, I use [GitHub](https://github.com/) for professional purposes and [GitLab](https://gitlab.com/) for personal projects.

        To follow this part of the course, I invite you to [create a GitHub account](https://github.com/join?source=header-home) or [GitLab account](https://gitlab.com/users/sign_in), your choice. Create a new repository (*New repository*).

        ### Basic Git Workflow

        1. **Initialize a repository**: Create a new repository on GitHub or GitLab
        2. **Clone the repository**: Download a local copy to your computer
        3. **Make changes**: Modify files in your project
        4. **Stage changes**: Select which changes to include in the next snapshot
        5. **Commit**: Create a snapshot of your changes with a descriptive message
        6. **Push**: Upload your commits to the online repository

        ### Using Git from the Command Line

        Here are the essential git commands:
        """
    )
    return


@app.cell
def __():
    # Essential git commands (run these in terminal, not Python)

    # Initialize a new git repository
    # git init

    # Clone an existing repository
    # git clone https://github.com/username/repository.git

    # Check status of your repository
    # git status

    # Stage changes for commit
    # git add filename.py        # Add specific file
    # git add data/*.csv         # Add all CSV files in data folder
    # git add .                  # Add all changes

    # Commit changes
    # git commit -m "Descriptive message about what changed"

    # Push changes to remote repository
    # git push origin main

    # Pull changes from remote repository
    # git pull origin main

    # View commit history
    # git log

    # Create a new branch
    # git checkout -b new-feature

    # Switch between branches
    # git checkout branch-name

    # Merge branches
    # git merge branch-name
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Using GitHub Desktop

        For those who prefer a graphical interface, [GitHub Desktop](https://desktop.github.com/) provides an easy way to work with git. It allows you to:

        - Clone repositories
        - View changes visually
        - Create commits with descriptive messages
        - Push and pull changes
        - Create and switch between branches
        - View commit history

        The interface is intuitive and helps you avoid common command-line mistakes.

        ### What to Include in Your Repository

        When sharing code on GitHub, include:

        - **README.md**: Describe your project, how to install dependencies, and how to run the code
        - **Data files**: Raw data in open formats (CSV, JSON, Parquet, SQLite)
        - **Code files**: Python scripts or Marimo notebooks (`.py` files)
        - **Requirements file**: `pyproject.toml` or `requirements.txt` listing dependencies
        - **License**: Choose an open source license (MIT, GPL, BSD, etc.)
        - **.gitignore**: List files/folders that shouldn't be tracked (see below)

        ### The .gitignore File

        Not everything should be tracked by git. Create a `.gitignore` file to exclude:

        ```
        # Python bytecode
        __pycache__/
        *.pyc
        *.pyo
        *.pyd

        # Virtual environments
        .venv/
        venv/
        env/

        # Jupyter notebook checkpoints
        .ipynb_checkpoints/

        # IDE settings
        .vscode/
        .idea/

        # OS files
        .DS_Store
        Thumbs.db

        # Large data files (consider using Git LFS instead)
        data/large_dataset.csv

        # Sensitive information
        .env
        credentials.json
        secrets.txt
        ```

        ### Sharing Your Work

        When publishing research, include a link to your GitHub repository in the methodology section. You can use a shortened link with [git.io](https://git.io/). For example:

        > The data and Python code used to compute the results are both available as supplementary material at https://github.com/username/project-name.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Managing Dependencies with uv

        While packages are continually updated, we need to ensure we know exactly which version was used for strict reproducibility. When reviewing articles, I ask that module names and version numbers be explicitly cited and referenced. For example, in an article on compositional analysis of lettuce inoculated with bacteria, I wrote:

        > Computations were performed in Python version 3.11. The main packages used in the data analysis workflow were pandas version 2.0.0 for data manipulation, lets-plot version 4.0.0 for visualization, statsmodels version 0.14.0 for statistical modeling, and scikit-learn version 1.3.0 for machine learning. The data and computations are publicly available at https://github.com/username/project-name.

        This way, anyone (colleagues, auditors, or yourself in the future) can reproduce the code by installing the cited versions. But this is tedious. That's why we use **uv**, a modern Python package manager.

        ### Why uv?

        **uv** is a fast, reliable Python package manager that:

        - Installs packages quickly (10-100x faster than pip)
        - Creates reproducible environments with lock files
        - Manages Python versions
        - Works seamlessly with virtual environments
        - Is a single binary with no dependencies

        ### Setting Up uv for Your Project

        Here's how to use uv for a reproducible project:
        """
    )
    return


@app.cell
def __():
    # Install uv (run in terminal)
    # curl -LsSf https://astral.sh/uv/install.sh | sh

    # Initialize a new project
    # uv init my-project
    # cd my-project

    # Install Python (if needed)
    # uv python install 3.11

    # Create a virtual environment
    # uv venv

    # Activate the virtual environment (Linux/Mac)
    # source .venv/bin/activate

    # Activate the virtual environment (Windows)
    # .venv\Scripts\activate

    # Add dependencies to your project
    # uv add pandas numpy scipy statsmodels scikit-learn lets-plot nuee marimo pymc

    # This creates/updates pyproject.toml and uv.lock

    # Install dependencies from lock file (for reproducibility)
    # uv sync

    # Run a script with uv
    # uv run python script.py

    # Run a Marimo notebook with uv
    # uv run marimo edit notebook.py
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### The pyproject.toml File

        When you use `uv add`, it updates your `pyproject.toml` file, which describes your project:

        ```toml
        [project]
        name = "bee-cranberry-analysis"
        version = "0.1.0"
        description = "Analysis of bee pollination in cranberry fields"
        requires-python = ">=3.11"
        dependencies = [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "statsmodels>=0.14.0",
            "scikit-learn>=1.3.0",
            "lets-plot>=4.0.0",
            "nuee>=0.1.0",
            "marimo>=0.1.0",
            "pymc>=5.0.0",
        ]
        ```

        ### The uv.lock File

        The `uv.lock` file contains exact versions of all dependencies and their sub-dependencies. This ensures anyone can recreate your exact environment:

        ```bash
        # Someone else can recreate your environment with:
        uv sync
        ```

        This is much more reliable than a simple `requirements.txt` because it locks all transitive dependencies.

        ### Workflow with Git and uv

        1. **Leto** (researcher 1) creates a project and adds dependencies with uv
        2. Leto commits `pyproject.toml` and `uv.lock` to git
        3. Leto pushes to GitHub
        4. **Ghanima** (researcher 2) clones the repository
        5. Ghanima runs `uv sync` to install exact same dependencies
        6. Ghanima can now reproduce Leto's results exactly

        If Leto updates dependencies later:

        1. Leto runs `uv add new-package` or `uv update existing-package`
        2. Leto commits the updated `uv.lock` file
        3. Leto pushes to GitHub
        4. Ghanima pulls the changes
        5. Ghanima runs `uv sync` to update their environment

        ### Including uv Files in Git

        Your `.gitignore` should track dependency files but not the virtual environment:

        ```
        # Track these
        pyproject.toml
        uv.lock

        # Don't track these
        .venv/
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## The Reprex: Reproducible Example

        When I discovered a bug in the **weathercan** package, I opened an issue on GitHub indicating the error message, hoping the bug's origin could be easily deduced. A developer asked me for a **reprex**. I was disappointed to learn that a reprex wasn't a species of dinosaur, but rather a **re**producible **ex**ample.

        > 📗 **Reprex**: A reproducible example.

        I tried to isolate the problem to reproduce the error with the minimum code possible. From code of more than 7,000 lines (these course notes), I arrived at this:
        """
    )
    return


@app.cell
def __(np, pd):
    # Example reprex: isolating a bug
    # Creating minimal reproducible example

    # Bad reprex: too much context, hard to debug
    # data = pd.read_csv("my_complicated_data.csv")
    # result = complicated_pipeline(data)  # Error somewhere!

    # Good reprex: minimal, self-contained
    import sys

    # Create minimal data that reproduces the issue
    _data_reprex = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Minimal code that reproduces the error
    # try:
    #     result = data['C']  # KeyError: 'C' not in columns
    # except KeyError as e:
    #     print(f"Error: {e}")

    # Include system information
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    return (sys,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Synthesizing a question isn't easy (creating this reproducible example took me nearly 2 hours). But answering an unsynthesized question is even harder. That's why you'll systematically be asked for a *reprex* when you post a question related to a systematic error, most often in programming.

        > A reproducible example allows someone to recreate the error you obtained simply by copy-pasting your code. - [Hadley Wickham](https://gist.github.com/hadley/270442)

        According to [Hadley Wickham](https://gist.github.com/hadley/270442) (R guru, but the principle applies to Python), a *reprex* should include four elements:

        1. **Load packages at the beginning of the code**.
        2. **Load data**, which can be example data or data included directly in the code (like randomly generated data).
        3. **Ensure your code is a minimal example** (remove the superfluous) and that it's easily readable.
        4. **Include the output of system information** (Python version, package versions, OS), which indicates the hardware and software platform where you generated the error. This is particularly important for bugs.

        When you think you've generated your *reprex*, restart Python (or your kernel), then run your code to ensure the error can be generated in a clean environment.

        ### Creating a Good Reprex

        Here's a template for creating reproducible examples:
        """
    )
    return


@app.cell
def __(np, pd, sys):
    # REPREX TEMPLATE
    # =================

    # 1. Import required libraries
    # import pandas as pd
    # import numpy as np

    # 2. Create minimal data
    # Use small, self-contained data that demonstrates the issue
    _minimal_data = pd.DataFrame(
        {"species": ["setosa", "virginica", "versicolor"] * 3, "measurement": np.random.randn(9)}
    )

    # 3. Minimal code that reproduces the issue
    # Keep only the essential code that triggers the error
    # try:
    #     result = minimal_data.groupby('species').apply(some_function)
    # except Exception as e:
    #     print(f"Error occurred: {e}")

    # 4. Include system information
    _reprex_info = {
        "Python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "Platform": sys.platform,
    }

    # Print system info
    print("System Information:")
    for _key, _value in _reprex_info.items():
        print(f"  {_key}: {_value}")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Where to Post Your Reprex

        Good places to ask for help with reproducible examples:

        - **Stack Overflow**: Tag your question with [python], [pandas], etc.
        - **GitHub Issues**: If you've found a bug in a specific package
        - **Reddit r/learnpython**: Friendly community for Python learners
        - **Mailing lists**: Many scientific Python packages have mailing lists

        Remember: the easier you make it for others to help you, the faster you'll get good answers!
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Summary

        In this chapter, we covered the principles of open and reproducible science:

        - **Open Science**: Five key aspects - methodology, data, code, peer review, and access
        - **Reproducible Code**: Following BES guidelines for organizing data and code
        - **Project Structure**: Organizing folders, files, and using relative paths
        - **Marimo Notebooks**: Creating reactive, reproducible notebooks stored as Python files
        - **Version Control**: Using git and GitHub to track changes and collaborate
        - **Dependency Management**: Using uv to create reproducible environments with `pyproject.toml` and `uv.lock`
        - **Reproducible Examples**: Creating minimal reprexes to get help and report bugs

        By following these practices, you'll make your research more transparent, reproducible, and valuable to the scientific community.

        ### Key Takeaways

        1. Always use open formats for data (CSV, JSON, Parquet, SQLite)
        2. Document your code thoroughly with comments and markdown
        3. Use version control (git) for all projects
        4. Lock your dependencies with uv for reproducibility
        5. Share your code and data on GitHub or GitLab
        6. Create minimal reproducible examples when asking for help

        ### Next Steps

        In the following chapters, we'll apply these reproducibility principles as we explore biostatistics, data analysis, and machine learning with Python. Every analysis will be contained in Marimo notebooks, version controlled with git, and have locked dependencies with uv.
        """
    )
    return


if __name__ == "__main__":
    app.run()
