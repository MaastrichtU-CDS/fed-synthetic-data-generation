"""
Test configuration and fixtures for fed-synthetic-data testing suite.

This module provides common fixtures and utilities used across all test modules.
It includes synthetic data generation and shared test utilities.
"""

import pytest
import warnings

import pandas as pd
import numpy as np

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
