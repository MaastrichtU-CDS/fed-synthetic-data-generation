# Federated Synthetic Data Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python library for generating synthetic data in a federated learning context with privacy-preserving capabilities.

## Purpose

This library provides tools and utilities for training synthetic data generators across multiple nodes in a federated manner, enabling:
- Privacy-preserving synthetic data generation
- Distributed training without centralizing data
- Differential privacy mechanisms
- Flexible aggregation strategies

## Features

- **Federated Training**: Train synthetic data generators across multiple nodes without sharing raw data
- **Privacy Preservation**: Built-in differential privacy mechanisms
- **Tabular Data Support**: Generate high-quality synthetic tabular data
- **Flexible Architecture**: Modular design for easy extension and customization
- **Comprehensive Testing**: Unit, integration, and empirical validation tests

## Installation

### From Source

```bash
git clone https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation.git
cd fed-synthetic-data-generation
pip install -e .
```

### Development Installation

For development with all testing dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from fed_synthetic_data import TabularSyntheticGenerator

# Load your data
data = pd.read_csv("your_data.csv")

# Create and train generator
generator = TabularSyntheticGenerator(
    model_type="tabular",
    privacy_mechanism="dp",
    epsilon=1.0
)
generator.fit(data)

# Generate synthetic data
synthetic_data = generator.sample(n_samples=1000)
```

### Federated Training

```python
from fed_synthetic_data import train_federated_generator, FederatedTrainer
from fed_synthetic_data.utils import split_data_by_nodes

# Simulate federated setup
node_datasets = split_data_by_nodes(data, n_nodes=3)

# Train on each node
generators = []
for node_data in node_datasets:
    gen = train_federated_generator(
        node_data,
        privacy_config={"privacy_mechanism": "dp", "epsilon": 1.0}
    )
    generators.append(gen)

# Use federated trainer for coordination
trainer = FederatedTrainer(
    aggregation_method="fedavg",
    num_rounds=10
)

for i, dataset in enumerate(node_datasets):
    trainer.add_node(f"node_{i}", dataset)

results = trainer.train()
```

### With Privacy Preservation

```python
from fed_synthetic_data.privacy_measures import (
    apply_differential_privacy,
    compute_privacy_budget
)

# Apply differential privacy
private_data = apply_differential_privacy(
    data,
    epsilon=1.0,
    mechanism="laplace"
)

# Compute privacy budget
total_budget = compute_privacy_budget(
    num_queries=10,
    epsilon_per_query=0.1,
    composition_method="basic"
)
```

## Library Structure

The library is organized into the following modules:

- **`synthetic_generator`**: Core synthetic data generation classes
- **`federated_training`**: Federated learning coordination and aggregation
- **`privacy_measures`**: Differential privacy and privacy-preserving mechanisms
- **`utils`**: Utility functions for data processing and validation

## Testing

The library includes comprehensive testing:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/empirical/     # Empirical validation

# Run with coverage
pytest --cov=fed_synthetic_data --cov-report=html
```

For detailed testing information, see the [Testing Guide](tests/TESTING_GUIDE.md).

## Documentation

- [Testing Guide](tests/TESTING_GUIDE.md): Comprehensive guide to the testing framework
- API documentation (coming soon)

## Architecture

The library follows a modular architecture inspired by:
- [v6-tools-general](https://github.com/STRONGAYA/v6-tools-general) for testing and structure
- [fed-mostlyai-engine](https://github.com/skarrea/fed-mostlyai-engine) for synthetic data generation concepts

### Key Components

1. **Synthetic Data Generators**: Create synthetic data that preserves statistical properties
2. **Federated Trainer**: Coordinate training across multiple nodes
3. **Privacy Mechanisms**: Apply differential privacy and other privacy-preserving techniques
4. **Utilities**: Helper functions for data processing and validation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Requirements

- Python >= 3.10
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- PyTorch >= 2.0.0

See `pyproject.toml` for complete dependency list.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This library's architecture is inspired by:
- [STRONGAYA/v6-tools-general](https://github.com/STRONGAYA/v6-tools-general) - Testing and library structure
- [skarrea/fed-mostlyai-engine](https://github.com/skarrea/fed-mostlyai-engine) - Synthetic data generation approach

## Contact

For questions or issues, please open an issue on GitHub or contact Maastricht University - CDS.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fed_synthetic_data,
  title = {Federated Synthetic Data Generation},
  author = {Maastricht University - CDS},
  year = {2024},
  url = {https://github.com/MaastrichtU-CDS/fed-synthetic-data-generation}
}
```
