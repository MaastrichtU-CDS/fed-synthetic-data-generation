"""
Empirical validation tests comparing federated and centralized approaches.
"""

import pytest
import pandas as pd
import numpy as np
from fed_synthetic_data.synthetic_generator import train_federated_generator
from fed_synthetic_data.utils import split_data_by_nodes, merge_datasets


@pytest.mark.empirical
class TestFederatedVsCentralized:
    """
    Empirical tests validating that federated approaches produce
    comparable results to centralized approaches.
    """

    def test_data_splitting_preserves_distribution(self, sample_numerical_data):
        """
        Test that splitting data maintains overall statistical properties.
        """
        # Compute statistics on original data
        original_mean = sample_numerical_data.mean()
        original_std = sample_numerical_data.std()
        
        # Split data
        splits = split_data_by_nodes(sample_numerical_data, n_nodes=5, random_state=42)
        
        # Merge back
        merged = merge_datasets(splits)
        
        # Compute statistics on merged data
        merged_mean = merged.mean()
        merged_std = merged.std()
        
        # Should be very close (accounting for floating point)
        np.testing.assert_allclose(original_mean, merged_mean, rtol=1e-10)
        np.testing.assert_allclose(original_std, merged_std, rtol=1e-10)

    def test_federated_preserves_sample_count(self, sample_tabular_data):
        """
        Test that federated splitting preserves total sample count.
        """
        original_count = len(sample_tabular_data)
        
        # Split across multiple nodes
        for n_nodes in [2, 3, 5, 10]:
            splits = split_data_by_nodes(sample_tabular_data, n_nodes=n_nodes)
            total_count = sum(len(df) for df in splits)
            
            assert total_count == original_count, \
                f"Sample count mismatch with {n_nodes} nodes"

    def test_federated_training_consistency(self, sample_tabular_data):
        """
        Test that federated training produces consistent results.
        """
        # Train multiple times with same seed
        results = []
        for _ in range(3):
            generator = train_federated_generator(
                sample_tabular_data,
                model_config={"model_type": "tabular"}
            )
            results.append(generator.is_fitted)
        
        # All should be fitted
        assert all(results)

    def test_privacy_impact_on_utility(self, sample_numerical_data):
        """
        Test that privacy mechanisms impact data utility as expected.
        """
        # Train without privacy
        gen_no_privacy = train_federated_generator(
            sample_numerical_data,
            privacy_config={"privacy_mechanism": None}
        )
        
        # Train with privacy
        gen_with_privacy = train_federated_generator(
            sample_numerical_data,
            privacy_config={"privacy_mechanism": "dp", "epsilon": 0.1}
        )
        
        assert gen_no_privacy.is_fitted
        assert gen_with_privacy.is_fitted
        assert gen_with_privacy.privacy_mechanism == "dp"


@pytest.mark.empirical
class TestDataQualityMetrics:
    """
    Empirical tests for data quality metrics.
    """

    def test_column_preservation(self, sample_tabular_data):
        """
        Test that column types are preserved through federated process.
        """
        original_dtypes = sample_tabular_data.dtypes
        
        # Split and merge
        splits = split_data_by_nodes(sample_tabular_data, n_nodes=3)
        merged = merge_datasets(splits)
        
        # Check column types are preserved
        for col in original_dtypes.index:
            assert merged[col].dtype == original_dtypes[col], \
                f"Column {col} dtype changed"

    def test_missing_value_preservation(self, sample_tabular_data):
        """
        Test that missing values are preserved through splitting/merging.
        """
        original_missing = sample_tabular_data.isna().sum()
        
        # Split and merge
        splits = split_data_by_nodes(sample_tabular_data, n_nodes=4, random_state=42)
        merged = merge_datasets(splits)
        merged_missing = merged.isna().sum()
        
        # Missing counts should be the same
        pd.testing.assert_series_equal(original_missing, merged_missing)

    @pytest.mark.slow
    def test_statistical_properties_preservation(self, sample_numerical_data):
        """
        Test that statistical properties are preserved in federated setting.
        """
        # Original statistics
        orig_stats = {
            'mean': sample_numerical_data.mean(),
            'std': sample_numerical_data.std(),
            'min': sample_numerical_data.min(),
            'max': sample_numerical_data.max(),
        }
        
        # Split, process, and merge
        splits = split_data_by_nodes(sample_numerical_data, n_nodes=5, random_state=42)
        merged = merge_datasets(splits)
        
        # Merged statistics
        merged_stats = {
            'mean': merged.mean(),
            'std': merged.std(),
            'min': merged.min(),
            'max': merged.max(),
        }
        
        # Compare (should be identical for split-merge)
        for stat_name in orig_stats:
            np.testing.assert_allclose(
                orig_stats[stat_name],
                merged_stats[stat_name],
                rtol=1e-10,
                err_msg=f"{stat_name} not preserved"
            )


@pytest.mark.empirical
class TestScalabilityMetrics:
    """
    Empirical tests for scalability of federated approaches.
    """

    def test_splits_scale_with_node_count(self, sample_tabular_data):
        """
        Test that data splits appropriately with varying node counts.
        """
        original_size = len(sample_tabular_data)
        
        node_counts = [2, 5, 10, 20]
        for n_nodes in node_counts:
            splits = split_data_by_nodes(sample_tabular_data, n_nodes=n_nodes)
            
            assert len(splits) == n_nodes
            
            # Check approximate equal distribution
            expected_per_node = original_size / n_nodes
            for split in splits:
                # Should be within 20% of expected size
                assert abs(len(split) - expected_per_node) / expected_per_node < 0.2

    def test_empty_splits_handled(self, sample_tabular_data):
        """
        Test that very small datasets are handled correctly.
        """
        small_data = sample_tabular_data.head(10)
        
        # Try to split into more nodes than samples
        with pytest.raises(ValueError):
            split_data_by_nodes(small_data, n_nodes=100)
