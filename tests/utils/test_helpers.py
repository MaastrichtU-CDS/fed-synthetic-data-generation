"""
Helper utilities for testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def generate_synthetic_tabular_data(
    n_samples: int = 1000,
    n_features: int = 5,
    n_categorical: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic tabular data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_categorical: Number of categorical features
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(random_state)
    
    n_numerical = n_features - n_categorical
    
    data = {}
    
    # Generate numerical features
    for i in range(n_numerical):
        data[f"num_feature_{i}"] = np.random.normal(50, 10, n_samples)
    
    # Generate categorical features
    for i in range(n_categorical):
        data[f"cat_feature_{i}"] = np.random.choice(
            ["A", "B", "C", "D"],
            n_samples
        )
    
    df = pd.DataFrame(data)
    
    # Convert categorical columns
    for col in [c for c in df.columns if c.startswith("cat_")]:
        df[col] = df[col].astype("category")
    
    return df


def compare_distributions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    numerical_tolerance: float = 0.1,
    categorical_tolerance: float = 0.05
) -> Dict[str, bool]:
    """
    Compare distributions between two DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        numerical_tolerance: Relative tolerance for numerical features
        categorical_tolerance: Tolerance for categorical distributions
        
    Returns:
        Dictionary with comparison results per column
    """
    results = {}
    
    # Check numerical columns
    num_cols = df1.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col in df2.columns:
            mean_diff = abs(df1[col].mean() - df2[col].mean())
            std_diff = abs(df1[col].std() - df2[col].std())
            
            mean_rel_diff = mean_diff / (df1[col].mean() + 1e-10)
            std_rel_diff = std_diff / (df1[col].std() + 1e-10)
            
            results[col] = (
                mean_rel_diff < numerical_tolerance and
                std_rel_diff < numerical_tolerance
            )
    
    # Check categorical columns
    cat_cols = df1.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col in df2.columns:
            # Compare value counts
            vc1 = df1[col].value_counts(normalize=True)
            vc2 = df2[col].value_counts(normalize=True)
            
            # Check if distributions are similar
            max_diff = 0
            for val in set(vc1.index) | set(vc2.index):
                diff = abs(vc1.get(val, 0) - vc2.get(val, 0))
                max_diff = max(max_diff, diff)
            
            results[col] = max_diff < categorical_tolerance
    
    return results


def assert_federated_equals_centralized(
    federated_result: Any,
    centralized_result: Any,
    rtol: float = 0.05,
    atol: float = 0.1
) -> None:
    """
    Assert that federated and centralized results are approximately equal.
    
    Args:
        federated_result: Result from federated computation
        centralized_result: Result from centralized computation
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    if isinstance(federated_result, (int, float)) and isinstance(centralized_result, (int, float)):
        np.testing.assert_allclose(
            federated_result,
            centralized_result,
            rtol=rtol,
            atol=atol
        )
    elif isinstance(federated_result, np.ndarray) and isinstance(centralized_result, np.ndarray):
        np.testing.assert_allclose(
            federated_result,
            centralized_result,
            rtol=rtol,
            atol=atol
        )
    elif isinstance(federated_result, dict) and isinstance(centralized_result, dict):
        assert set(federated_result.keys()) == set(centralized_result.keys())
        for key in federated_result:
            assert_federated_equals_centralized(
                federated_result[key],
                centralized_result[key],
                rtol=rtol,
                atol=atol
            )
    else:
        assert federated_result == centralized_result


def create_mock_federated_setup(
    data: pd.DataFrame,
    n_nodes: int = 3
) -> List[Dict[str, Any]]:
    """
    Create a mock federated learning setup for testing.
    
    Args:
        data: Data to split across nodes
        n_nodes: Number of nodes to simulate
        
    Returns:
        List of node configurations
    """
    from fed_synthetic_data.utils import split_data_by_nodes
    
    splits = split_data_by_nodes(data, n_nodes=n_nodes, random_state=42)
    
    nodes = []
    for i, node_data in enumerate(splits):
        node_config = {
            "node_id": f"node_{i}",
            "data": node_data,
            "data_size": len(node_data),
            "features": list(node_data.columns),
        }
        nodes.append(node_config)
    
    return nodes


def validate_synthetic_data_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    min_quality_score: float = 0.7
) -> Dict[str, Any]:
    """
    Validate the quality of synthetic data.
    
    Args:
        real_data: Original real data
        synthetic_data: Generated synthetic data
        min_quality_score: Minimum acceptable quality score
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        "shape_match": real_data.shape[1] == synthetic_data.shape[1],
        "columns_match": set(real_data.columns) == set(synthetic_data.columns),
        "has_samples": len(synthetic_data) > 0,
    }
    
    # Compare distributions
    if quality_metrics["columns_match"]:
        distribution_comparison = compare_distributions(real_data, synthetic_data)
        quality_metrics["distribution_similarity"] = distribution_comparison
        
        # Compute overall quality score
        if distribution_comparison:
            quality_score = sum(distribution_comparison.values()) / len(distribution_comparison)
            quality_metrics["quality_score"] = quality_score
            quality_metrics["passes_threshold"] = quality_score >= min_quality_score
        else:
            quality_metrics["quality_score"] = 0.0
            quality_metrics["passes_threshold"] = False
    else:
        quality_metrics["quality_score"] = 0.0
        quality_metrics["passes_threshold"] = False
    
    return quality_metrics


def compute_privacy_metrics(
    epsilon: float,
    delta: float,
    num_queries: int
) -> Dict[str, float]:
    """
    Compute privacy metrics for evaluation.
    
    Args:
        epsilon: Privacy budget per query
        delta: Delta parameter
        num_queries: Number of queries
        
    Returns:
        Dictionary with privacy metrics
    """
    from fed_synthetic_data.privacy_measures import compute_privacy_budget
    
    total_epsilon = compute_privacy_budget(num_queries, epsilon)
    
    return {
        "epsilon": epsilon,
        "delta": delta,
        "num_queries": num_queries,
        "total_epsilon": total_epsilon,
        "privacy_loss": total_epsilon,
    }
