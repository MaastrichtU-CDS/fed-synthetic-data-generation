"""
Integration tests for federated synthetic data generation workflows.
"""

import pytest
import pandas as pd
import numpy as np
from fed_synthetic_data import (
    TabularSyntheticGenerator,
    FederatedTrainer,
    train_federated_generator,
)
from fed_synthetic_data.utils import split_data_by_nodes, merge_datasets


@pytest.mark.integration
class TestFederatedTrainingWorkflow:
    """Integration tests for complete federated training workflows."""

    def test_basic_federated_workflow(self, sample_tabular_data):
        """Test a basic end-to-end federated training workflow."""
        # Split data across nodes
        node_datasets = split_data_by_nodes(sample_tabular_data, n_nodes=3)
        
        # Train local generators
        local_generators = []
        for node_data in node_datasets:
            generator = train_federated_generator(node_data)
            local_generators.append(generator)
        
        # Verify all generators are trained
        assert len(local_generators) == 3
        assert all(gen.is_fitted for gen in local_generators)

    def test_federated_trainer_with_nodes(self, federated_datasets):
        """Test FederatedTrainer with multiple nodes."""
        trainer = FederatedTrainer(num_rounds=5, aggregation_method="fedavg")
        
        # Add nodes
        for i, dataset in enumerate(federated_datasets):
            trainer.add_node(f"node_{i}", dataset)
        
        # Train
        results = trainer.train()
        
        assert results["rounds_completed"] == 5
        assert len(results["results"]) == 5

    def test_privacy_preserved_training(self, sample_tabular_data):
        """Test federated training with privacy preservation."""
        node_datasets = split_data_by_nodes(sample_tabular_data, n_nodes=2)
        
        # Train with privacy - use correct parameter names
        privacy_config = {"privacy_mechanism": "dp", "epsilon": 1.0}
        generators = []
        for node_data in node_datasets:
            generator = train_federated_generator(
                node_data,
                privacy_config=privacy_config
            )
            generators.append(generator)
        
        # Verify privacy settings are applied
        assert all(gen.privacy_mechanism is not None for gen in generators)


@pytest.mark.integration
class TestDataProcessingPipeline:
    """Integration tests for data processing pipelines."""

    def test_split_and_merge_preserves_data(self, sample_tabular_data):
        """Test that splitting and merging preserves data integrity."""
        original_len = len(sample_tabular_data)
        original_cols = set(sample_tabular_data.columns)
        
        # Split
        splits = split_data_by_nodes(sample_tabular_data, n_nodes=4, random_state=42)
        
        # Merge back
        merged = merge_datasets(splits)
        
        assert len(merged) == original_len
        assert set(merged.columns) == original_cols

    def test_processing_with_missing_data(self, sample_tabular_data):
        """Test processing pipeline with missing data."""
        # Data already has missing values
        splits = split_data_by_nodes(sample_tabular_data, n_nodes=3)
        
        # Each split should have some data
        assert all(len(df) > 0 for df in splits)
        
        # Merge should preserve missing values
        merged = merge_datasets(splits)
        assert merged.isna().any().any()


@pytest.mark.integration
class TestGeneratorIntegration:
    """Integration tests for synthetic generator workflows."""

    def test_fit_and_sample_workflow(self, sample_tabular_data):
        """Test complete fit and sample workflow."""
        generator = TabularSyntheticGenerator(model_type="tabular")
        
        # Fit
        generator.fit(sample_tabular_data)
        
        # Sample
        synthetic_data = generator.sample(n_samples=100)
        
        assert isinstance(synthetic_data, pd.DataFrame)

    def test_evaluation_workflow(self, sample_tabular_data):
        """Test evaluation of generated data."""
        generator = TabularSyntheticGenerator()
        generator.fit(sample_tabular_data)
        
        synthetic_data = generator.sample(n_samples=500)
        
        # Evaluate
        metrics = generator.evaluate(sample_tabular_data, synthetic_data)
        
        assert isinstance(metrics, dict)


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end integration tests for realistic scenarios."""

    def test_multi_node_training_and_generation(self, sample_tabular_data):
        """Test multi-node training and synthetic data generation."""
        # Simulate federated setting
        n_nodes = 4
        node_datasets = split_data_by_nodes(
            sample_tabular_data,
            n_nodes=n_nodes,
            method="iid",
            random_state=42
        )
        
        # Train on each node
        node_generators = []
        for i, node_data in enumerate(node_datasets):
            generator = TabularSyntheticGenerator(model_type="tabular")
            generator.fit(node_data)
            node_generators.append(generator)
        
        # Generate synthetic data from each node
        synthetic_datasets = []
        for generator in node_generators:
            synthetic = generator.sample(n_samples=100)
            synthetic_datasets.append(synthetic)
        
        # Verify outputs
        assert len(synthetic_datasets) == n_nodes
        assert all(isinstance(df, pd.DataFrame) for df in synthetic_datasets)

    @pytest.mark.slow
    def test_large_scale_federated_training(self, sample_tabular_data):
        """Test federated training with larger dataset and more nodes."""
        # Create a larger dataset
        large_data = pd.concat([sample_tabular_data] * 5, ignore_index=True)
        
        # Split across many nodes
        n_nodes = 10
        node_datasets = split_data_by_nodes(large_data, n_nodes=n_nodes)
        
        # Train federated
        trainer = FederatedTrainer(num_rounds=3, aggregation_method="fedavg")
        
        for i, dataset in enumerate(node_datasets):
            trainer.add_node(f"node_{i}", dataset)
        
        results = trainer.train()
        
        assert results["rounds_completed"] == 3
        assert len(trainer.nodes) == n_nodes
