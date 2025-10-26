# nuee Documentation

This directory contains the Sphinx documentation for the nuee package.

## Building the Documentation

### Prerequisites

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Or install just the documentation requirements:

```bash
pip install sphinx sphinx-rtd-theme
```

### Building HTML Documentation

From the `docs` directory, run:

```bash
make html
```

The generated HTML files will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the documentation.

### Other Formats

```bash
make latexpdf  # Build PDF (requires LaTeX)
make epub      # Build EPUB
make linkcheck # Check for broken links
make clean     # Remove build files
```

## Documentation Structure

- `conf.py`: Sphinx configuration
- `index.rst`: Main documentation page
- `installation.rst`: Installation instructions
- `quickstart.rst`: Quick start guide
- `user_guide.rst`: Detailed user guide
- `examples.rst`: Example usage
- `api_reference.rst`: API reference (auto-generated from docstrings)
- `api/`: Individual module API documentation

## Writing Docstrings

This project uses NumPy-style docstrings. Example:

```python
def function(param1, param2):
    """
    Brief description of function.

    Longer description providing more details.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2.

    Returns
    -------
    type
        Description of return value.

    Examples
    --------
    >>> function(1, 2)
    3

    See Also
    --------
    other_function : Related function

    Notes
    -----
    Additional notes about the function.

    References
    ----------
    .. [1] Author (Year). Title. Journal.
    """
    pass
```

## Hosting Documentation

The documentation can be hosted on:
- **Read the Docs**: Connect your GitHub repository to automatically build and host docs
- **GitHub Pages**: Use GitHub Actions to build and deploy
- **Any web server**: Copy the `_build/html` directory

### Read the Docs Setup

1. Sign up at https://readthedocs.org/
2. Import your GitHub repository
3. RTD will automatically build docs on each commit

### GitHub Pages

Add a GitHub Actions workflow to build and deploy:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.x
      - run: pip install -e ".[dev]"
      - run: cd docs && make html
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```
