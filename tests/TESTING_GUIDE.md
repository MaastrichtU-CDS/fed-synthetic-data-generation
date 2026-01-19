# Testing Guide

This document provides comprehensive guidance for testing the `fed-synthetic-data` library.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)

## Overview

The testing framework ensures the reliability and correctness of all functionality, especially in federated scenarios. Tests are organized into several categories, each serving a specific purpose.

## Test Structure

```
tests/
├── conftest.py                    # Common fixtures and test utilities
├── unit/                          # Unit tests for individual functions
│   ├── test_synthetic_generator.py
│   ├── test_federated_training.py
│   ├── test_privacy_measures.py
│   └── test_utils.py
├── integration/                   # Integration tests
│   └── test_federated_workflow.py
├── empirical/                     # Empirical validation tests
│   └── test_federated_validation.py
└── utils/                         # Test helper utilities
    └── test_helpers.py
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -e ".[test]"
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run empirical tests only
pytest tests/empirical/

# Run with coverage report
pytest --cov=fed_synthetic_data --cov-report=html

# Run specific test module
pytest tests/unit/test_synthetic_generator.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "privacy"

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation. These tests:
- Are fast and deterministic
- Use mocks for external dependencies
- Focus on a single unit of functionality
- Should have high coverage

Example:
```python
@pytest.mark.unit
def test_generator_initialization():
    generator = TabularSyntheticGenerator()
    assert not generator.is_fitted
```

### Integration Tests

Test complete workflows and component interactions. These tests:
- Verify that components work together correctly
- Test realistic usage scenarios
- May take longer to run
- Use real (test) data and minimal mocking

Example:
```python
@pytest.mark.integration
def test_federated_training_workflow(sample_data):
    splits = split_data_by_nodes(sample_data, n_nodes=3)
    generators = [train_federated_generator(split) for split in splits]
    assert all(gen.is_fitted for gen in generators)
```

### Empirical Tests

Validate that federated approaches produce comparable results to centralized approaches. These tests:
- Compare federated vs. centralized computation results
- Verify mathematical equivalence
- Test privacy-utility tradeoffs
- May allow for small tolerances due to federated computation

Example:
```python
@pytest.mark.empirical
def test_federated_preserves_statistics(data):
    splits = split_data_by_nodes(data, n_nodes=5)
    merged = merge_datasets(splits)
    assert_distributions_equal(data, merged)
```

### Performance Tests

Benchmark function performance with varying data sizes (marked as `slow`):
```python
@pytest.mark.slow
@pytest.mark.performance
def test_large_scale_training(large_dataset):
    # Performance test with large dataset
    pass
```

## Writing Tests

### Test Guidelines

1. **Use descriptive test names** that explain what is being tested
2. **Include both positive and negative test cases**
3. **Test edge cases and error conditions**
4. **Use realistic synthetic data** from fixtures
5. **Mock external dependencies** appropriately
6. **Validate both structure and values** of results

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_with_sample_data(sample_tabular_data):
    """sample_tabular_data is automatically provided by pytest."""
    assert len(sample_tabular_data) == 1000
```

Available fixtures:
- `sample_tabular_data`: Mixed numerical and categorical data
- `sample_numerical_data`: Numerical data only
- `sample_categorical_data`: Categorical data only
- `federated_datasets`: Multiple datasets simulating nodes
- `edge_case_data`: Edge cases for robust testing
- `privacy_config`: Privacy configuration parameters
- `federated_config`: Federated learning configuration

### Custom Assertions

Use custom assertion helpers from `conftest.py`:

```python
from tests.conftest import assert_dataframes_equal, assert_privacy_preserved

def test_custom_assertions(data1, data2):
    assert_dataframes_equal(data1, data2, rtol=1e-5)
```

### Example Test Structure

```python
import pytest
from fed_synthetic_data import TabularSyntheticGenerator

@pytest.mark.unit
class TestGeneratorFeature:
    """Group related tests in a class."""
    
    def test_positive_case(self):
        """Test normal operation."""
        generator = TabularSyntheticGenerator()
        # Test implementation
        assert generator is not None
    
    def test_negative_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            # Test code that should raise
            pass
    
    def test_edge_case(self):
        """Test boundary conditions."""
        # Test implementation
        pass
```

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.empirical      # Empirical validation
@pytest.mark.slow           # Slow test
@pytest.mark.performance    # Performance benchmark
```

Run specific marker categories:
```bash
pytest -m unit              # Run only unit tests
pytest -m "not slow"        # Skip slow tests
pytest -m "integration or empirical"  # Run integration or empirical
```

## Continuous Integration

Tests run automatically on:
- Every push to the repository
- Every pull request
- Multiple Python versions (3.10+)

CI includes:
- Running all tests
- Code coverage reporting
- Linting and type checking
- Security scanning

## Coverage Requirements

Aim for:
- **Unit tests**: >90% coverage
- **Integration tests**: Cover all major workflows
- **Empirical tests**: Validate key federated vs. centralized comparisons

Check coverage:
```bash
pytest --cov=fed_synthetic_data --cov-report=html
# Open htmlcov/index.html in browser
```

## Debugging Tests

### Run with more verbosity
```bash
pytest -vv                  # Very verbose
pytest --tb=short          # Shorter traceback
pytest --tb=long           # Longer traceback
pytest -s                   # Show print statements
```

### Run specific test
```bash
pytest tests/unit/test_synthetic_generator.py::TestTabularSyntheticGenerator::test_initialization
```

### Drop into debugger on failure
```bash
pytest --pdb               # Drop into pdb on failure
pytest --trace             # Drop into pdb at start of test
```

## Contributing Tests

When adding new functionality:

1. **Add unit tests** for all new functions/methods
2. **Add integration tests** for complete workflows
3. **Add empirical tests** if relevant for federated vs. centralized
4. **Update this guide** if adding new test categories
5. **Ensure tests pass** before submitting PR

### Test Checklist

- [ ] Unit tests added with good coverage
- [ ] Integration tests for workflows added
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Tests are deterministic (use random seeds)
- [ ] Tests are documented
- [ ] All tests pass locally

## Known Issues

Some empirical tests may occasionally show small numerical differences due to:
- Floating point precision in federated computations
- Random sampling in privacy mechanisms
- Platform-specific numerical libraries

These are expected and within acceptable tolerances.

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
