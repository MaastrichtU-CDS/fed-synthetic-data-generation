"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
import pandas as pd
from fed_synthetic_data.utils import (
    safe_log,
    validate_data,
    check_column_types,
    compute_data_statistics,
    split_data_by_nodes,
    merge_datasets,
    safe_divide,
    normalize_weights,
)


@pytest.mark.unit
class TestSafeLog:
    """Test cases for safe_log function."""

    def test_logs_info_message(self):
        """Test logging an info message."""
        # Should not raise any errors
        safe_log("info", "Test message")

    def test_logs_different_levels(self):
        """Test logging with different levels."""
        levels = ["debug", "info", "warning", "error", "critical"]
        
        for level in levels:
            safe_log(level, f"Test {level} message")

    def test_handles_invalid_level(self):
        """Test that invalid level falls back to info."""
        # Should not raise, should use info level
        safe_log("invalid_level", "Test message")


@pytest.mark.unit
class TestValidateData:
    """Test cases for validate_data function."""

    def test_validates_correct_data(self, sample_tabular_data):
        """Test validation of correct data."""
        result = validate_data(sample_tabular_data)
        assert result is True

    def test_raises_on_non_dataframe(self):
        """Test that non-DataFrame raises error."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            validate_data([1, 2, 3])

    def test_raises_on_insufficient_rows(self, sample_tabular_data):
        """Test that insufficient rows raises error."""
        with pytest.raises(ValueError, match="at least.*rows"):
            validate_data(sample_tabular_data, min_rows=10000)

    def test_raises_on_missing_columns(self, sample_tabular_data):
        """Test that missing required columns raises error."""
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(sample_tabular_data, required_columns=["nonexistent_col"])

    def test_validates_with_required_columns(self, sample_tabular_data):
        """Test validation with required columns."""
        result = validate_data(
            sample_tabular_data,
            required_columns=["age", "gender"]
        )
        assert result is True


@pytest.mark.unit
class TestCheckColumnTypes:
    """Test cases for check_column_types function."""

    def test_auto_detects_types(self, sample_tabular_data):
        """Test automatic type detection."""
        result = check_column_types(sample_tabular_data)
        
        assert isinstance(result, dict)
        assert "categorical" in result
        assert "numerical" in result
        assert "n_categorical" in result
        assert "n_numerical" in result

    def test_uses_provided_columns(self, sample_tabular_data):
        """Test with explicitly provided column lists."""
        result = check_column_types(
            sample_tabular_data,
            categorical_columns=["gender"],
            numerical_columns=["age", "height"]
        )
        
        assert "gender" in result["categorical"]
        assert "age" in result["numerical"]
        assert "height" in result["numerical"]


@pytest.mark.unit
class TestComputeDataStatistics:
    """Test cases for compute_data_statistics function."""

    def test_computes_basic_statistics(self, sample_tabular_data):
        """Test computation of basic statistics."""
        stats = compute_data_statistics(sample_tabular_data)
        
        assert isinstance(stats, dict)
        assert "n_rows" in stats
        assert "n_columns" in stats
        assert "n_missing" in stats
        assert "memory_usage_mb" in stats

    def test_correct_row_count(self, sample_tabular_data):
        """Test that row count is correct."""
        stats = compute_data_statistics(sample_tabular_data)
        assert stats["n_rows"] == len(sample_tabular_data)

    def test_correct_column_count(self, sample_tabular_data):
        """Test that column count is correct."""
        stats = compute_data_statistics(sample_tabular_data)
        assert stats["n_columns"] == len(sample_tabular_data.columns)


@pytest.mark.unit
class TestSplitDataByNodes:
    """Test cases for split_data_by_nodes function."""

    def test_basic_split(self, sample_tabular_data):
        """Test basic data splitting."""
        splits = split_data_by_nodes(sample_tabular_data, n_nodes=3)
        
        assert len(splits) == 3
        assert all(isinstance(df, pd.DataFrame) for df in splits)

    def test_split_preserves_data(self, sample_tabular_data):
        """Test that split preserves all data."""
        splits = split_data_by_nodes(sample_tabular_data, n_nodes=4)
        
        total_rows = sum(len(df) for df in splits)
        assert total_rows == len(sample_tabular_data)

    def test_iid_split(self, sample_tabular_data):
        """Test IID splitting method."""
        splits = split_data_by_nodes(
            sample_tabular_data,
            n_nodes=3,
            method="iid",
            random_state=42
        )
        
        assert len(splits) == 3

    def test_non_iid_split(self, sample_tabular_data):
        """Test non-IID splitting method."""
        splits = split_data_by_nodes(
            sample_tabular_data,
            n_nodes=3,
            method="non-iid"
        )
        
        assert len(splits) == 3

    def test_raises_on_invalid_nodes(self, sample_tabular_data):
        """Test that invalid node count raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            split_data_by_nodes(sample_tabular_data, n_nodes=0)
        
        with pytest.raises(ValueError, match="cannot exceed"):
            split_data_by_nodes(sample_tabular_data, n_nodes=100000)

    def test_raises_on_unknown_method(self, sample_tabular_data):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown splitting method"):
            split_data_by_nodes(sample_tabular_data, n_nodes=3, method="unknown")


@pytest.mark.unit
class TestMergeDatasets:
    """Test cases for merge_datasets function."""

    def test_basic_merge(self, federated_datasets):
        """Test basic dataset merging."""
        merged = merge_datasets(federated_datasets)
        
        assert isinstance(merged, pd.DataFrame)
        
        total_expected = sum(len(df) for df in federated_datasets)
        assert len(merged) == total_expected

    def test_raises_on_empty_list(self):
        """Test that empty dataset list raises error."""
        with pytest.raises(ValueError, match="No datasets"):
            merge_datasets([])

    def test_merge_single_dataset(self, sample_tabular_data):
        """Test merging a single dataset."""
        merged = merge_datasets([sample_tabular_data])
        assert len(merged) == len(sample_tabular_data)


@pytest.mark.unit
class TestSafeDivide:
    """Test cases for safe_divide function."""

    def test_normal_division(self):
        """Test normal division."""
        result = safe_divide(10.0, 2.0)
        assert result == 5.0

    def test_division_by_zero(self):
        """Test division by zero returns default."""
        result = safe_divide(10.0, 0.0, default=0.0)
        assert result == 0.0

    def test_custom_default(self):
        """Test custom default value."""
        result = safe_divide(10.0, 0.0, default=99.0)
        assert result == 99.0

    def test_handles_nan(self):
        """Test handling of NaN."""
        result = safe_divide(10.0, np.nan, default=0.0)
        assert result == 0.0

    def test_handles_inf(self):
        """Test handling of infinity."""
        result = safe_divide(10.0, np.inf, default=0.0)
        assert result == 0.0


@pytest.mark.unit
class TestNormalizeWeights:
    """Test cases for normalize_weights function."""

    def test_normalizes_weights(self):
        """Test basic weight normalization."""
        weights = [1.0, 2.0, 3.0]
        normalized = normalize_weights(weights)
        
        assert len(normalized) == 3
        assert sum(normalized) == pytest.approx(1.0)

    def test_already_normalized(self):
        """Test weights that are already normalized."""
        weights = [0.25, 0.25, 0.5]
        normalized = normalize_weights(weights)
        
        assert sum(normalized) == pytest.approx(1.0)

    def test_handles_zero_sum(self):
        """Test handling of zero sum (returns equal weights)."""
        weights = [0.0, 0.0, 0.0]
        normalized = normalize_weights(weights)
        
        assert all(w == pytest.approx(1.0 / 3.0) for w in normalized)

    def test_raises_on_empty_list(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_weights([])
