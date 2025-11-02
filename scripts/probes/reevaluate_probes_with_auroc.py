#!/usr/bin/env python3
"""Re-evaluate existing trained probes with AUROC scores.

This script loads already-trained probes and re-evaluates them on the test data
to compute AUROC scores without needing to retrain. It updates the probe_results.csv
files with the new AUROC column.

Usage:
    python scripts/probes/reevaluate_probes_with_auroc.py --base_dir outputs/probes
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError as e:
    print(f"Error: Required dependencies not available: {e}")
    print("Please install scikit-learn")
    sys.exit(1)

from src.ign_inf_unlearning.models.probes import LinearProbe, MLPProbe


def load_activations_from_pickle(file_path: str) -> np.ndarray:
    """Load activations from a pickle file."""
    with open(file_path, 'rb') as f:
        activations = pickle.load(f)
    
    # Handle list of mean activations
    if isinstance(activations, list):
        numpy_activations = []
        for activation in activations:
            if hasattr(activation, 'cpu'):  # PyTorch tensor
                activation = activation.cpu().numpy()
            elif hasattr(activation, 'numpy'):  # TensorFlow tensor
                activation = activation.numpy()
            elif not isinstance(activation, np.ndarray):
                activation = np.array(activation)
            numpy_activations.append(activation)
        activations = np.vstack(numpy_activations)
    
    return activations


def load_dataset_activations(
    data_dir: str, 
    layer_names: List[str]
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Load activations and metadata for a dataset."""
    data_path = Path(data_dir)
    
    # Try to find dataset in model-specific subdirectory first
    dataset_csv = None
    model_dirs = list(data_path.glob("*/"))
    if model_dirs:
        model_dir = model_dirs[0]
        dataset_csv = model_dir / "dataset.csv"
    else:
        dataset_csv = data_path / "dataset.csv"
        if not dataset_csv.exists():
            dataset_files = list(data_path.glob("dataset_*.csv"))
            if dataset_files:
                dataset_csv = dataset_files[0]
    
    if not dataset_csv or not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found in {data_path}")
    
    metadata_df = pd.read_csv(dataset_csv)
    
    # Load activations for each layer
    activations_dict = {}
    for layer_name in layer_names:
        layer_file = None
        if model_dirs:
            layer_file = model_dirs[0] / f"{layer_name}.pkl"
        else:
            layer_file = data_path / f"{layer_name}.pkl"
            if not layer_file.exists():
                layer_files = list(data_path.glob(f"{layer_name}_*.pkl"))
                if layer_files:
                    layer_file = layer_files[0]
        
        if layer_file and layer_file.exists():
            activations_dict[layer_name] = load_activations_from_pickle(str(layer_file))
        else:
            print(f"Warning: Layer file not found: {layer_file}")
    
    return activations_dict, metadata_df


def prepare_binary_dataset(
    datasets: List[str],
    data_base_dir: str,
    layer_names: List[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Prepare binary classification dataset from multiple datasets."""
    all_activations = {layer: [] for layer in layer_names}
    all_labels = []
    
    for label, dataset_name in enumerate(datasets):
        dataset_dir = os.path.join(data_base_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        print(f"Loading dataset: {dataset_name} (label: {label})")
        activations_dict, metadata_df = load_dataset_activations(dataset_dir, layer_names)
        
        # Get the number of samples
        n_samples = None
        for layer_name in layer_names:
            if layer_name in activations_dict:
                n_samples = len(activations_dict[layer_name])
                break
        
        if n_samples is None:
            raise ValueError(f"No valid activations found for dataset {dataset_name}")
        
        # Add activations for each layer
        for layer_name in layer_names:
            if layer_name in activations_dict:
                all_activations[layer_name].append(activations_dict[layer_name])
            else:
                placeholder = np.zeros((n_samples, 1))
                all_activations[layer_name].append(placeholder)
        
        # Add labels
        labels = np.full(n_samples, label)
        all_labels.append(labels)
    
    # Concatenate all data
    combined_activations = {}
    for layer_name in layer_names:
        if all_activations[layer_name]:
            combined_activations[layer_name] = np.vstack(all_activations[layer_name])
    
    combined_labels = np.concatenate(all_labels)
    
    return combined_activations, combined_labels


def reevaluate_probe_with_auroc(
    probe_path: str,
    probe_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Load a trained probe and evaluate it with AUROC."""
    
    if not os.path.exists(probe_path):
        print(f"Warning: Probe file not found: {probe_path}")
        return None
    
    # Determine device (use CPU if CUDA not available)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the probe with appropriate device mapping
    if probe_type == 'linear':
        probe = LinearProbe.load_probe(probe_path, map_location=device)
    elif probe_type == 'mlp':
        probe = MLPProbe.load_probe(probe_path, map_location=device)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    # Evaluate
    metrics, _, probabilities = probe.evaluate(X_test, y_test)
    
    # Compute F1 score
    y_pred = np.argmax(probabilities, axis=1)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Compute AUROC
    auroc = roc_auc_score(y_test, probabilities[:, 1])
    
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': f1,
        'auroc': auroc
    }


def reevaluate_probe_directory(
    probe_dir: str,
    data_base_dir: str,
    datasets: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """Re-evaluate all probes in a directory and update results."""
    
    probe_path = Path(probe_dir)
    
    # Check if probe_results.csv exists
    results_csv = probe_path / "probe_results.csv"
    if not results_csv.exists():
        print(f"No probe_results.csv found in {probe_dir}")
        return None
    
    # Load existing results
    print(f"\nProcessing: {probe_dir}")
    existing_df = pd.read_csv(results_csv)
    
    # Check if AUROC already exists
    if 'auroc' in existing_df.columns:
        print("AUROC already present, skipping...")
        return existing_df
    
    # Extract layer names from the results
    layer_names = existing_df['layer'].unique()
    
    # Filter out averaged_layers for data loading
    data_layer_names = [l for l in layer_names if l != 'averaged_layers']
    
    # Load activations
    print(f"Loading datasets: {datasets}")
    try:
        activations_dict, labels = prepare_binary_dataset(
            datasets, data_base_dir, data_layer_names
        )
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None
    
    # Create train/test split with same random state
    first_layer = next(iter(activations_dict.keys()))
    n_samples = len(activations_dict[first_layer])
    indices = np.arange(n_samples)
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    y_test = labels[test_indices]
    
    # Re-evaluate each probe and add AUROC
    new_results = []
    
    for _, row in existing_df.iterrows():
        layer_name = row['layer']
        probe_type = row['probe_type']
        
        # Skip averaged_layers for now (would need special handling)
        if layer_name == 'averaged_layers':
            print(f"Skipping {layer_name} (averaged layers)")
            new_row = row.to_dict()
            new_row['auroc'] = np.nan
            new_results.append(new_row)
            continue
        
        # Get test data for this layer
        if layer_name not in activations_dict:
            print(f"Warning: No activations for {layer_name}")
            new_row = row.to_dict()
            new_row['auroc'] = np.nan
            new_results.append(new_row)
            continue
        
        X_test = activations_dict[layer_name][test_indices]
        
        # Load and evaluate probe
        probe_file = f"{probe_type}_probe.pt"
        layer_dir = probe_path / layer_name
        probe_file_path = layer_dir / probe_file
        
        print(f"Re-evaluating {layer_name} / {probe_type}")
        metrics = reevaluate_probe_with_auroc(
            str(probe_file_path),
            probe_type,
            X_test,
            y_test
        )
        
        if metrics:
            new_row = row.to_dict()
            new_row['auroc'] = metrics['auroc']
            print(f"  AUROC: {metrics['auroc']:.4f}")
        else:
            new_row = row.to_dict()
            new_row['auroc'] = np.nan
        
        new_results.append(new_row)
    
    # Create new DataFrame with AUROC
    new_df = pd.DataFrame(new_results)
    
    # Save updated results
    new_df.to_csv(results_csv, index=False)
    print(f"Updated results saved to: {results_csv}")
    
    return new_df


def find_available_datasets(data_base_dir: str) -> List[str]:
    """Find all available dataset directories."""
    data_path = Path(data_base_dir)
    datasets = []
    
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir():
            datasets.append(dataset_dir.name)
    
    return sorted(datasets)


def infer_datasets_from_probe_dir(probe_dir_name: str, available_datasets: List[str]) -> List[str]:
    """Infer which datasets were used based on directory name and available datasets.
    
    This uses a greedy matching approach to find the two dataset names that
    best match the probe directory name.
    """
    # Try to find matching datasets
    matched_datasets = []
    remaining_name = probe_dir_name
    
    # Sort datasets by length (longest first) to match longer names first
    sorted_datasets = sorted(available_datasets, key=len, reverse=True)
    
    for dataset in sorted_datasets:
        if dataset in remaining_name:
            matched_datasets.append(dataset)
            # Remove the matched dataset from the name to avoid duplicate matches
            remaining_name = remaining_name.replace(dataset, '', 1)
            
            if len(matched_datasets) == 2:
                break
    
    return matched_datasets


def find_probe_directories(base_dir: str, data_base_dir: str) -> List[Tuple[str, List[str]]]:
    """Find all probe directories and infer dataset names."""
    base_path = Path(base_dir)
    probe_dirs = []
    
    # Get available datasets
    available_datasets = find_available_datasets(data_base_dir)
    print(f"Available datasets: {available_datasets}")
    
    # Look for model directories
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Look for dataset directories within each model
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            results_csv = dataset_dir / "probe_results.csv"
            if not results_csv.exists():
                continue
            
            # Infer dataset names from directory name
            probe_dir_name = dataset_dir.name
            datasets = infer_datasets_from_probe_dir(probe_dir_name, available_datasets)
            
            if len(datasets) == 2:
                probe_dirs.append((str(dataset_dir), datasets))
                print(f"Mapped {probe_dir_name} -> {datasets}")
            else:
                print(f"Warning: Could not infer 2 datasets from directory name: {probe_dir_name}")
                print(f"  Found: {datasets}")
    
    return probe_dirs


def main():
    """Main function to re-evaluate probes with AUROC."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing trained probes with AUROC scores."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="outputs/probes",
        help="Base directory containing probe subdirectories"
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="data/activations",
        help="Base directory containing activation datasets"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test size (must match original training)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state (must match original training)"
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        help="Specific probe directory to process (optional)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        help="Dataset names if processing a specific probe_dir"
    )
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="Show what would be processed without actually doing it"
    )
    
    args = parser.parse_args()
    
    if args.probe_dir:
        # Process a specific directory
        if not args.datasets:
            print("Error: --datasets required when using --probe_dir")
            sys.exit(1)
        
        reevaluate_probe_directory(
            args.probe_dir,
            args.data_base_dir,
            args.datasets,
            args.test_size,
            args.random_state
        )
    else:
        # Find and process all probe directories
        print(f"Searching for probe directories in: {args.base_dir}")
        probe_dirs = find_probe_directories(args.base_dir, args.data_base_dir)
        
        if not probe_dirs:
            print("No probe directories found!")
            sys.exit(1)
        
        print(f"\nFound {len(probe_dirs)} probe directories\n")
        
        if args.dry_run:
            print("DRY RUN - Would process the following:")
            for probe_dir, datasets in probe_dirs:
                print(f"  {probe_dir}")
                print(f"    Datasets: {datasets}")
            print("\nRun without --dry_run to actually process these directories.")
        else:
            for probe_dir, datasets in probe_dirs:
                reevaluate_probe_directory(
                    probe_dir,
                    args.data_base_dir,
                    datasets,
                    args.test_size,
                    args.random_state
                )
    
    print("\nRe-evaluation complete!")


if __name__ == "__main__":
    main()

