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
- **GitHub Pages**: Automatically deployed via GitHub Actions (recommended)
- **Read the Docs**: Connect your GitHub repository to automatically build and host docs
- **Any web server**: Copy the `_build/html` directory

### GitHub Pages (Automated)

The repository includes a GitHub Actions workflow (`.github/workflows/docs.yml`) that automatically builds and deploys documentation to GitHub Pages when:
- A new version tag is pushed (e.g., `v0.1.0`)
- Changes are pushed to the `main` branch
- Manually triggered via the Actions tab

**To enable GitHub Pages:**

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Build and deployment":
   - Source: Select **GitHub Actions**
4. Push a tag or commit to `main` to trigger the deployment:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

The documentation will be available at: `https://<username>.github.io/<repository>/`

For this repository: `https://essicolo.github.io/nuee/`

### Read the Docs Setup

The repository includes a `.readthedocs.yml` configuration file.

1. Sign up at https://readthedocs.org/
2. Import your GitHub repository
3. RTD will automatically build docs on each commit

The documentation will be available at: `https://<project>.readthedocs.io/`
