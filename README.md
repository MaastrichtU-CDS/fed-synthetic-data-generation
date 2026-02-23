# STRONG AYA's General Vantage6 tools

<p align="center">
<a href="https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation/workflows/"><img alt="Test status" src="https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation/workflows/Test%20Suite/badge.svg"></a>
<a><img alt="Coverage" src="https://raw.githubusercontent.com/MaastrichtU-CDS/fed-synthetic-data-generation/main/tests/coverage-badge.svg"></a>
<a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img alt="Licence: Apache 2.0" src="https://img.shields.io/badge/Licence-Apache%202.0-blue.svg"></a>
<br>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://flake8.pycqa.org/"><img alt="Linting: flake8" src="https://img.shields.io/badge/linting-flake8-informational"></a>
<a href="http://mypy-lang.org/"><img alt="Type checking: mypy" src="https://img.shields.io/badge/type%20checking-mypy-informational"></a>
<a href="https://github.com/PyCQA/bandit"><img alt="Security: bandit" src="https://img.shields.io/badge/security-bandit-informational"></a>
<a href="https://github.com/pyupio/safety"><img alt="Security: safety" src="https://img.shields.io/badge/security-safety-informational"></a>
</p>

# Purpose of this repository

This repository contains general functionalities and tools for the STRONG AYA project.
They are designed to be used with the Vantage6 framework for federated analytics and learning
and are intended to facilitate and simplify the development of Vantage6 algorithms.

The code in this repository is available as a Python library here on GitHub or through direct reference with `pip`.

# Structure of the repository

The various functions are organised in different sections, consisting of:

-

# Usage

The library provides functions that can be included in a federated algorithm as the algorithm developer sees fit.
The functions are designed to be modular and can be used independently or in combination with other functions.

## Including the library in a Vantage6 algorithm

The library can be included in your Vantage6 algorithm by listing it in the `requirements.txt` and `setup.py` file of
your
algorithm.

For the `requirements.txt` file, you can add the following line to the file:

```
git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git@v0.1.0
```

For the `setup.py` file, you can add the following line to the `install_requires` list:

```python
        "fed-synthetic-data-generation @ git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git@v0.1.0",
```

The algorithm's `setup.py`, particularly the `install_requirements`, section file should then look something like this:

```python
from os import path
from codecs import open
from setuptools import setup, find_packages

# We are using a README.md, if you do not have this in your folder, simply replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name="v6-not-an-actual-algorithm",
    version="1.0.0",
    description="Fictive Vantage6 algorithm that performs federated synthetic data generator training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/STRONGAYA/v6-not-an-actual-algorithm",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vantage6-algorithm-tools",
        "numpy",
        "pandas",
        "fed-synthetic-data-generation @ git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git@v0.1.0"
        # other dependencies
    ]
)
```

## Central (aggregating) example

Example usage of various functions in a central (aggregating) section of a federated algorithm:

```python

```

## Node or local (participating) example

Example usage of various functions in a node (participating) section of a Vantage6 algorithm:

```python

```

The various functions are available through `pip install` for debugging and testing purposes.
The library can be installed as follows:

```bash
pip install git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git
```

# Testing

This repository includes a comprehensive testing framework to ensure the reliability and correctness of all functions,
especially in federated scenarios.

## Test Structure

```
tests/
├── conftest.py                           # Common fixtures and test utilities
├── unit/                                 # Unit tests for individual functions
│   ├── test_federated_training.py        # Tests for federated training functions
│   ├── test_privacy_measures.py          # Tests for privacy functions
│   └── test_utils.py                      # Tests for utility functions 
├── empirical/                            # Empirical validation tests
│   └── test_federated_vs_centralised.py  # Federated vs centralised comparisons
└── utils/                                # Test helper utilities
    └── test_helpers.py                   # Validation and comparison tools
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock hypothesis faker
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run empirical tests only
pytest tests/empirical/

# Run with coverage report
pytest --cov=vantage6_strongaya_general --cov-report=html

# Run specific test module
pytest tests/unit/test_general_statistics.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Test individual functions in isolation
- **Empirical Tests**: Validate federated vs centralised mathematical equivalence
- **Performance Tests**: Benchmark function performance with large datasets
- **Edge Case Tests**: Test behaviour with unusual data distributions

### Federated vs Centralised Validation

The test suite includes comprehensive empirical validation that federated statistical computations produce equivalent
results to their centralised counterparts:

```python

```

### Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions:

- Multiple Python versions (starting with 3.10)
- Code coverage reporting
- Performance benchmarking
- Security scanning

## Contributing to Tests

When contributing new functionality:

1. **Add unit tests** for all new functions
2. **Add integration tests** for complete workflows
3. **Add empirical tests** for federated vs centralised scenarios
3. **Include edge case testing** for robustness
4. **Update test data** if needed for new scenarios
5. **Maintain an acceptable degree of code coverage**

### Test Guidelines

- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use realistic synthetic data
- Mock external dependencies (AlgorithmClient, environment variables)
- Validate both structure and values of results

# Contributers

- J. Hogenboom
- B. S. Abrahamsen

# References

- [Vantage6](vantage6.ai)