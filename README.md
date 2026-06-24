# Tools for federated synthetic data generation

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

This repository provides helper functions for federated synthetic data generator training, e.g. using
[fed-mostlyai-engine](https://github.com/skarrea/fed-mostlyai-engine). The functions are designed to be imported
into a federated algorithm, e.g. [Vantage6](https://vantage6.ai), and cover weight aggregation, serialisation for JSON
transport, and data preparation utilities.

Additionally, this lightweight library can be used to compose a model using the federated synthetic data generator
training weights (after it has been trained).

The code is available as a Python library directly from GitHub or via `pip`.

# Structure of the repository

The various functions are organised in different sections, consisting of:

src/fed_synthetic_data/ 
├── federated_training.py # Weight aggregation, JSON serialisation, loss evaluation, learning rate scheduling, early stopping
├── utils.py # Data preparation (column ordering), weight serialisation/deserialisation
├── post_training.py # Model weight loading and saving
└── privacy_measures.py # Privacy enhancing measures (planned)


# Usage

The library provides functions that can be included in a federated algorithm as the algorithm developer sees fit.
The functions are designed to be modular and can be used independently or in combination with other functions.

## Including the library in a Vantage6 algorithm

The library can be included in your Vantage6 algorithm by listing it in the `requirements.txt` and `pyproject.toml`
(or `setup.py`) of your algorithm.

For the `requirements.txt` file, you can add the following line:

```
git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git@v0.1.0
```

For a `setup.py`-based algorithm, add the following to the `install_requires` list:

```python
        "fed-synthetic-data-generation @ git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git@v0.1.0",
```

A minimal `setup.py` for a consuming algorithm might look like:

```python
from os import path
from codecs import open
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name="v6-not-an-actual-algorithm",
    version="1.0.0",
    description="Fictive Vantage6 algorithm that performs federated synthetic data generator training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/your-algorithm",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vantage6-algorithm-tools",
        "numpy",
        "pandas",
        "mostlyai-engine>=2.4.0",
        "fed-synthetic-data-generation @ git+https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git@version",
        # other dependencies
    ]
)
```

## Central (aggregating) example

Example usage of various functions in a central (aggregating) section of a federated algorithm:

```python
from fed_synthetic_data.federated_training import (
    aggregation_model_weights_weighted_average,
    weights_from_json,
    weights_to_json,
)

# results_from_nodes: list of JSON weight payloads + row counts returned by each node
aggregated_weights = aggregation_model_weights_weighted_average([
    (weights_from_json(node_result["weights"]), node_result["n_rows"])
    for node_result in results_from_nodes
])

# Serialise aggregated weights to send back to nodes
payload = weights_to_json(aggregated_weights)


```

## Node or local (participating) example

Example usage of various functions in a node (participating) section of a Vantage6 algorithm:

```python
from fed_synthetic_data.federated_training import weights_to_json
from fed_synthetic_data.utils import sort_columns

# Enforce a consistent column order before training
df = sort_columns(df, column_order=agreed_column_order)

# After local training, extract and serialise weights for the central node
weights = model.get_weights()
return {"weights": weights_to_json(weights), "n_rows": len(df)}
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
│   ├── test_post_training.py             # Tests for post-training functions
│   ├── test_utils.py                     # Tests for utility functions
│   └── test_imports.py                   # Tests for package imports
├── integration/                          # Integration tests
│   ├── test_post_training.py             # Integration tests for post-training
│   ├── test_serialisation_pipeline.py   # Tests for serialisation pipeline
│   └── test_utils_integration.py         # Integration tests for utilities
└── empirical/                            # Empirical validation tests
    ├── test_learning_rate_schedulers.py  # Tests for LR schedulers
    ├── test_mathematical_correctness.py # Mathematical validation tests
    └── test_pipeline_functionality.py   # Pipeline functionality tests
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
pytest --cov=fed_synthetic_data --cov-report=html

# Run specific test module
pytest tests/unit/test_federated_training.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Test individual functions in isolation
- **Performance Tests**: Benchmark function performance with large datasets
- **Edge Case Tests**: Test behaviour with unusual data distributions


# Contributors

- J. Hogenboom
- B. S. Abrahamsen

# References

- [Vantage6](https://vantage6.ai)