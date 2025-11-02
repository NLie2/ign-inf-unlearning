"""
Centralized path management for the IGN-INF-Unlearning project.

This module provides base paths across scripts, notebooks, and modules,
regardless of where they are run from.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Main data directories
DATA_PATH = PROJECT_ROOT / "data"
DATASETS_PATH = DATA_PATH / "datasets"
ACTIVATIONS_PATH = DATA_PATH / "activations"
RESULTS_PATH = PROJECT_ROOT / "results"

# Environment
ENV_FILE = PROJECT_ROOT / ".env"


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {DATA_PATH}")
    print(f"Datasets path: {DATASETS_PATH}")
    print(f"Activations path: {ACTIVATIONS_PATH}")
    print(f"Results path: {RESULTS_PATH}")
    print(f"Environment file: {ENV_FILE}")

