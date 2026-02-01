# Contributing to Deep Markov Model

Thank you for your interest in contributing to the Deep Markov Model project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dmm.git
   cd dmm
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/SilviaCleva/dmm.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- conda or virtualenv (recommended)

### Installation

1. Create a virtual environment:
   ```bash
   conda create -n dmm-dev python=3.10
   conda activate dmm-dev
   ```

2. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check dmm/

# Run type checker
mypy dmm/
```

## Code Style

### General Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Keep functions focused and concise
- Write self-documenting code

### Type Hints

All code must include type hints:

```python
from typing import List, Tuple, Optional
from torch import Tensor

def process_sequence(
    sequence: Tensor,
    normalize: bool = True,
) -> Tuple[Tensor, Optional[float]]:
    """Process a sequence tensor.

    Args:
        sequence: Input tensor of shape (T, features).
        normalize: Whether to normalize the output.

    Returns:
        Tuple of processed tensor and optional scale factor.
    """
    ...
```

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def train_epoch(self, sequences: List[Tensor]) -> float:
    """Train for one epoch over all sequences.

    Args:
        sequences: List of training sequence tensors,
            each of shape (T, features).

    Returns:
        Average ELBO loss for the epoch.

    Raises:
        ValueError: If sequences list is empty.

    Example:
        >>> trainer = DMMTrainer(model)
        >>> loss = trainer.train_epoch(train_seqs)
    """
```

### Formatting

We use `ruff` for linting and formatting:

```bash
# Check for issues
ruff check dmm/

# Auto-fix issues
ruff check dmm/ --fix

# Format code
ruff format dmm/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dmm --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching a pattern
pytest tests/ -k "test_transition" -v

# Skip slow tests
pytest tests/ -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures from `conftest.py` for common setup

Example test:

```python
import pytest
import torch
from dmm import ModelConfig, DMMFactory

class TestGatedTransition:
    """Tests for GatedTransition component."""

    def test_output_shape(self, model_config: ModelConfig) -> None:
        """Test that output has correct shape."""
        from dmm import GatedTransition

        trans = GatedTransition(
            z_dim=model_config.z_dim,
            transition_dim=model_config.transition_dim,
        )

        z_prev = torch.randn(4, model_config.z_dim)
        loc, scale = trans(z_prev)

        assert loc.shape == (4, model_config.z_dim)
        assert scale.shape == (4, model_config.z_dim)

    def test_scale_positive(self, model_config: ModelConfig) -> None:
        """Test that scale is always positive."""
        from dmm import GatedTransition

        trans = GatedTransition(
            z_dim=model_config.z_dim,
            transition_dim=model_config.transition_dim,
        )

        z_prev = torch.randn(4, model_config.z_dim)
        _, scale = trans(z_prev)

        assert (scale > 0).all()
```

### Test Coverage

Aim for at least 80% test coverage on new code. Check coverage with:

```bash
pytest tests/ --cov=dmm --cov-report=term-missing
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

2. Run linting and type checking:
   ```bash
   ruff check dmm/
   mypy dmm/
   ```

3. Update documentation if needed

4. Add tests for new functionality

### Submitting

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

### PR Guidelines

- Use a clear, descriptive title
- Reference any related issues
- Describe what changes were made and why
- Include screenshots for UI changes
- Ensure CI checks pass

### Review Process

1. A maintainer will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- Python version and OS
- Package versions (`pip freeze`)
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full error traceback

### Feature Requests

For feature requests, describe:

- The problem you're trying to solve
- Your proposed solution
- Alternative approaches considered
- Whether you're willing to implement it

## Questions?

If you have questions about contributing, feel free to:

- Open a GitHub Discussion
- Create an issue with the "question" label

Thank you for contributing!
