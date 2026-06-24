"""
Package import smoke tests.
Ensure all public modules are importable and the package __init__ doesn't break.
"""


def test_package_import():
    """The top-level package imports without error."""
    import fed_synthetic_data


def test_module_imports():
    """All public submodules are importable."""
    from fed_synthetic_data import federated_training
    from fed_synthetic_data import utils
    from fed_synthetic_data import post_training
    from fed_synthetic_data import privacy_measures


def test_public_api_symbols():
    """Public API symbols are accessible from their modules."""
    from fed_synthetic_data.federated_training import (
        aggregation_model_weights_weighted_average,
        evaluate_loss,
        should_stop_early,
    )
    from fed_synthetic_data.utils import (
        weights_to_json,
        weights_from_json,
        sort_columns,
    )
    from fed_synthetic_data.post_training import (
        load_weights_into_model,
        load_model_from_json_weights,
        save_model,
        save_model_weights,
    )
