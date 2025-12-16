# Contributing to nuee

Thank you for your interest in contributing to nuee! We welcome contributions from everyone and are committed to fostering an inclusive, respectful, and collaborative community.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, identity, or level of experience. We expect all participants to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

Harassment, discrimination, or exclusionary behavior of any kind will not be tolerated.

## How to Contribute

### Reporting Issues

If you find a bug, have a feature request, or want to suggest improvements:

1. Check existing [GitHub Issues](https://github.com/essicolo/nuee/issues) to avoid duplicates
2. Create a new issue with:
   - A clear, descriptive title
   - A detailed description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (Python version, OS, nuee version)
   - Minimal reproducible example (if applicable)

### Submitting Pull Requests

We welcome pull requests for bug fixes, new features, documentation improvements, and more.

#### Before You Start

1. Create or comment on an issue to discuss your proposed changes
2. Fork the repository
3. Create a new branch from `main` with a descriptive name:
   - `fix/description` for bug fixes
   - `feature/description` for new features
   - `docs/description` for documentation

#### Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nuee.git
   cd nuee
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests to ensure your setup works:
   ```bash
   pytest
   ```

#### Making Changes

1. Write clear, maintainable code that follows existing patterns
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass: `pytest`
5. Follow PEP 8 style guidelines (use a linter like `ruff` or `flake8`)

#### Commit Messages

Write clear commit messages that explain what changed and why:
```
Fix betadisper: change default permutations from 0 to 999

The betadisper function was returning p_value=nan because the default
value of the permutations parameter was 0, which skipped the
permutation test.
```

#### Submitting Your PR

1. Push your branch to your fork
2. Open a pull request against the `main` branch
3. Provide a clear description of:
   - What the PR does
   - Why the change is needed
   - Any related issues (use "Fixes #123" to auto-close issues)
   - Testing you've performed

4. Be responsive to review feedback
5. Update your PR as needed

### Testing

All contributions should include appropriate tests:

- Unit tests for individual functions
- Integration tests for complex workflows
- Ensure tests pass locally before submitting

Run tests with:
```bash
pytest
pytest tests/test_specific_module.py  # Run specific test file
```

### Documentation

Good documentation helps everyone:

- Update docstrings for any modified functions
- Follow NumPy docstring format
- Update relevant `.rst` files in `docs/` if needed
- Provide examples for new features
- Keep the user guide up to date

Build documentation locally:
```bash
cd docs
make html
```

### Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will be included in the next release

## Development Guidelines

### Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Write clear variable and function names
- Keep functions focused and modular
- Add comments for complex logic

### Compatibility

- Support Python 3.8+
- Maintain compatibility with major dependencies (NumPy, pandas, SciPy)
- Avoid breaking changes when possible
- Document any breaking changes clearly

### Performance

- Consider computational efficiency for large datasets
- Profile code for performance-critical sections
- Document any performance trade-offs

## Questions?

If you have questions about contributing, please:

- Open an issue on GitHub with the "question" label
- Check existing documentation and issues first

## Recognition

All contributors will be acknowledged in the project. We value every contribution, whether it's code, documentation, bug reports, or feature suggestions.

Thank you for helping make nuee better!
