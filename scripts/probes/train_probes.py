#!/usr/bin/env python3
"""Train and evaluate probes on model activations.

This script trains linear and MLP probes on activations extracted from different datasets
to detect separable concepts in the model's internal representations.

Usage:
    python scripts/probes/train_probes.py --datasets real_words_sciency gibberish \
                                         --model_name meta-llama/Llama-3.2-3B-Instruct \
                                         --layers 0 6 12 18 24 30 \
                                         --probe_types linear mlp
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
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: Some optional dependencies not available: {e}")
    print("Please install scikit-learn, matplotlib, and seaborn for full functionality")

from src.ign_inf_unlearning.models.probes import LinearProbe, MLPProbe


def load_activations_from_pickle(file_path: str) -> np.ndarray:
    """Load activations from a pickle file containing a list of mean activations.
    
    Args:
        file_path: Path to the pickle file containing activations
        
    Returns:
        Numpy array of mean activations (samples x features)
    """
    with open(file_path, 'rb') as f:
        activations = pickle.load(f)
    
    print(f"Raw activations type: {type(activations)}")
    
    # Handle list of mean activations
    if isinstance(activations, list):
        print(f"List with {len(activations)} elements")
        
        # Convert each element to numpy array and stack them
        numpy_activations = []
        for i, activation in enumerate(activations):
            if hasattr(activation, 'cpu'):  # PyTorch tensor
                activation = activation.cpu().numpy()
            elif hasattr(activation, 'numpy'):  # TensorFlow tensor
                activation = activation.numpy()
            elif not isinstance(activation, np.ndarray):  # Other types
                activation = np.array(activation)
            
            numpy_activations.append(activation)
        
        # Stack all activations into a single array
        activations = np.vstack(numpy_activations)
        print(f"Stacked activations shape: {activations.shape}")
    
    elif isinstance(activations, np.ndarray):
        print(f"Already numpy array: {activations.shape}")
    
    print(f"Final activations shape: {activations.shape}")
    return activations


def load_dataset_activations(
    data_dir: str, 
    layer_names: List[str]
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Load activations and metadata for a dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        layer_names: List of layer names to load (e.g., ['layer_0', 'layer_6'])
        
    Returns:
        Tuple of (activations_dict, metadata_df)
    """
    data_path = Path(data_dir)
    
    # Try to find dataset in model-specific subdirectory first
    dataset_csv = None
    model_dirs = list(data_path.glob("*/"))  # Look for subdirectories
    if model_dirs:
        # Use the first model subdirectory found
        model_dir = model_dirs[0]
        dataset_csv = model_dir / "dataset.csv"
        print(f"Using model subdirectory: {model_dir}")
    else:
        # Fallback to old format in root directory
        dataset_csv = data_path / "dataset.csv"
        if not dataset_csv.exists():
            # Try to find dataset file with model name pattern
            dataset_files = list(data_path.glob("dataset_*.csv"))
            if dataset_files:
                dataset_csv = dataset_files[0]  # Use the first one found
                print(f"Using dataset file: {dataset_csv}")
    
    if not dataset_csv or not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found in {data_path}")
    
    metadata_df = pd.read_csv(dataset_csv)
    
    # Load activations for each layer
    activations_dict = {}
    for layer_name in layer_names:
        # Try model subdirectory first
        layer_file = None
        if model_dirs:
            layer_file = model_dirs[0] / f"{layer_name}.pkl"
        else:
            # Fallback to old format
            layer_file = data_path / f"{layer_name}.pkl"
            if not layer_file.exists():
                # Try to find layer file with model name pattern
                layer_files = list(data_path.glob(f"{layer_name}_*.pkl"))
                if layer_files:
                    layer_file = layer_files[0]  # Use the first one found
                    print(f"Using layer file: {layer_file}")
        
        if layer_file and layer_file.exists():
            activations_dict[layer_name] = load_activations_from_pickle(str(layer_file))
            print(f"Loaded {layer_name}: {activations_dict[layer_name].shape}")
        else:
            print(f"Warning: Layer file not found: {layer_file}")
    
    return activations_dict, metadata_df


def prepare_binary_dataset(
    datasets: List[str],
    data_base_dir: str,
    layer_names: List[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Prepare binary classification dataset from multiple datasets.
    
    Args:
        datasets: List of dataset names (directories)
        data_base_dir: Base directory containing dataset folders
        layer_names: List of layer names to load
        
    Returns:
        Tuple of (combined_activations_dict, labels)
    """
    all_activations = {layer: [] for layer in layer_names}
    all_labels = []
    
    for label, dataset_name in enumerate(datasets):
        dataset_dir = os.path.join(data_base_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        print(f"\nLoading dataset: {dataset_name} (label: {label})")
        activations_dict, metadata_df = load_dataset_activations(dataset_dir, layer_names)
        
        # Get the number of samples from the first available layer
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
                # Create placeholder with zeros if layer is missing
                print(f"Warning: Missing {layer_name} for {dataset_name}, using zeros")
                placeholder = np.zeros((n_samples, 1))  # Will be handled later
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
    
    print("\nCombined dataset:")
    print(f"Total samples: {len(combined_labels)}")
    print(f"Label distribution: {np.bincount(combined_labels)}")
    
    return combined_activations, combined_labels


def train_and_evaluate_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe_types: List[str],
    layer_name: str,
    output_dir: str,
    num_epochs: int = 100
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate probes for a given layer.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        probe_types: List of probe types to train
        layer_name: Name of the layer
        output_dir: Directory to save trained probes
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary of results for each probe type
    """
    results = {}
    input_dim = X_train.shape[1]
    
    # Create output directory for this layer
    layer_output_dir = os.path.join(output_dir, layer_name)
    os.makedirs(layer_output_dir, exist_ok=True)
    
    print(f"\n--- Training probes for {layer_name} ---")
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Linear Probe
    if 'linear' in probe_types:
        print("\nTraining Linear Probe...")
        linear_probe = LinearProbe(input_size=input_dim)
        linear_probe.train_probe(X_train, y_train, num_epochs=num_epochs)
        
        metrics, _, probabilities = linear_probe.evaluate(X_test, y_test)
        
        # Compute F1 score
        y_pred = np.argmax(probabilities, axis=1)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Compute AUROC using probabilities for class 1
        auroc = roc_auc_score(y_test, probabilities[:, 1])
        
        results['linear'] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': f1,
            'auroc': auroc
        }
        
        print(f"Linear Probe - Accuracy: {metrics['accuracy']:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
        
        # Save probe
        probe_path = os.path.join(layer_output_dir, 'linear_probe.pt')
        linear_probe.save_probe(probe_path)
    
    # Train MLP Probe
    if 'mlp' in probe_types:
        print("\nTraining MLP Probe...")
        mlp_probe = MLPProbe(input_size=input_dim, hidden_size=64)
        mlp_probe.train_probe(X_train, y_train, num_epochs=num_epochs, lr=0.01)
        
        metrics, _, probabilities = mlp_probe.evaluate(X_test, y_test)
        
        # Compute F1 score
        y_pred = np.argmax(probabilities, axis=1)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Compute AUROC using probabilities for class 1
        auroc = roc_auc_score(y_test, probabilities[:, 1])
        
        results['mlp'] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': f1,
            'auroc': auroc
        }
        
        print(f"MLP Probe - Accuracy: {metrics['accuracy']:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
        
        # Save probe
        probe_path = os.path.join(layer_output_dir, 'mlp_probe.pt')
        mlp_probe.save_probe(probe_path)
    
    return results


def compute_averaged_activations(
    activations_dict: Dict[str, np.ndarray],
    layer_names: List[str]
) -> np.ndarray:
    """Compute averaged activations across multiple layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation arrays (already token-averaged)
        layer_names: List of layer names to average
        
    Returns:
        Averaged activations array across layers
    """
    valid_activations = []
    
    for layer_name in layer_names:
        if layer_name in activations_dict:
            valid_activations.append(activations_dict[layer_name])
    
    if not valid_activations:
        raise ValueError("No valid activations found for averaging")
    
    # Average across layers (each activation is already token-averaged)
    averaged = np.mean(valid_activations, axis=0)
    print(f"Averaged activations shape: {averaged.shape}")
    return averaged


def plot_results(
    results_df: pd.DataFrame,
    output_dir: str,
    datasets: List[str],
    model_name: str
) -> None:
    """Plot and save probe results.
    
    Args:
        results_df: DataFrame containing results
        output_dir: Directory to save plots
        datasets: List of dataset names used
        model_name: Name of the model
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract layer numbers for proper ordering
    def extract_layer_number(layer_name):
        if layer_name == 'averaged_layers':
            return 999  # Put averaged at the end
        return int(layer_name.split('_')[1])
    
    # Sort layers numerically
    results_df = results_df.copy()
    results_df['layer_num'] = results_df['layer'].apply(extract_layer_number)
    results_df = results_df.sort_values('layer_num')
    
    # Set up the plot style
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot accuracy as line plot
    accuracy_pivot = results_df.pivot(index='layer', columns='probe_type', values='accuracy')
    accuracy_pivot.plot(kind='line', ax=ax1, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Probe Accuracy by Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Layer')
    ax1.legend(title='Probe Type')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot F1 score as line plot
    f1_pivot = results_df.pivot(index='layer', columns='probe_type', values='f1_score')
    f1_pivot.plot(kind='line', ax=ax2, marker='o', linewidth=2, markersize=6)
    ax2.set_title('Probe F1-Score by Layer')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Layer')
    ax2.legend(title='Probe Type')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot AUROC as line plot
    if 'auroc' in results_df.columns:
        auroc_pivot = results_df.pivot(index='layer', columns='probe_type', values='auroc')
        auroc_pivot.plot(kind='line', ax=ax3, marker='o', linewidth=2, markersize=6)
        ax3.set_title('Probe AUROC by Layer')
        ax3.set_ylabel('AUROC')
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Layer')
        ax3.legend(title='Probe Type')
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # Add overall title
    dataset_str = ' vs '.join(datasets)
    safe_model_name = model_name.replace('/', '_')
    fig.suptitle(f'Probe Performance: {dataset_str}\nModel: {safe_model_name}', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"probe_results_{safe_model_name}_{'-'.join(datasets)}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def main():
    """Main function to train and evaluate probes."""
    parser = argparse.ArgumentParser(description="Train and evaluate probes on model activations.")
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs='+', 
        required=True,
        help="Names of datasets to use (e.g., 'real_words_sciency' 'gibberish')"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Name of the model used for naming output directories"
    )
    parser.add_argument(
        "--data_base_dir", 
        type=str, 
        default="data/activations",
        help="Base directory containing activation datasets"
    )
    parser.add_argument(
        "--output_base_dir", 
        type=str, 
        default="outputs/probes",
        help="Base directory to save trained probes and results"
    )
    parser.add_argument(
        "--layers", 
        type=str, 
        nargs='+',
        help="Layer numbers to train probes on (e.g., '0' '6' '12'). If not specified, will auto-detect."
    )
    parser.add_argument(
        "--probe_types", 
        type=str, 
        nargs='+', 
        default=['linear', 'mlp'],
        choices=['linear', 'mlp'],
        help="Types of probes to train"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=100,
        help="Number of epochs for training probes"
    )
    parser.add_argument(
        "--include_averaged", 
        action='store_true',
        help="Also train probes on averaged activations across all layers"
    )
    
    args = parser.parse_args()
    
    # Validate datasets exist
    for dataset in args.datasets:
        dataset_path = os.path.join(args.data_base_dir, dataset)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset directory not found: {dataset_path}")
            sys.exit(1)
    
    # Auto-detect layers if not specified
    if args.layers is None:
        # Check first dataset for available layers
        first_dataset_path = os.path.join(args.data_base_dir, args.datasets[0])
        dataset_path = Path(first_dataset_path)
        
        # Look for layer files in model subdirectory first
        layer_files = []
        model_dirs = list(dataset_path.glob("*/"))  # Look for subdirectories
        if model_dirs:
            # Use the first model subdirectory found
            model_dir = model_dirs[0]
            layer_files = list(model_dir.glob("layer_*.pkl"))
            print(f"Looking for layers in model subdirectory: {model_dir}")
        else:
            # Fallback to old format in root directory
            layer_files = list(dataset_path.glob("layer_*.pkl"))
            print(f"Looking for layers in root directory: {dataset_path}")
        
        if not layer_files:
            print(f"Error: No layer files found in {first_dataset_path}")
            sys.exit(1)
        
        # Extract layer numbers
        layer_numbers = []
        for layer_file in layer_files:
            layer_name = layer_file.stem  # e.g., "layer_0"
            if layer_name.startswith("layer_"):
                layer_numbers.append(layer_name.split("_")[1])
        
        layer_numbers.sort(key=int)
        args.layers = layer_numbers
        print(f"Auto-detected layers: {args.layers}")
    
    # Create layer names
    layer_names = [f"layer_{layer}" for layer in args.layers]
    
    # Create output directory
    safe_model_name = args.model_name.replace('/', '_')
    dataset_suffix = '_'.join(args.datasets)
    output_dir = os.path.join(args.output_base_dir, safe_model_name, dataset_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Training probes on datasets: {args.datasets}")
    print(f"Using layers: {layer_names}")
    print(f"Probe types: {args.probe_types}")
    
    # Load and prepare data
    print("\nLoading datasets...")
    print("Note: Activations will be averaged across tokens for each sample")
    activations_dict, labels = prepare_binary_dataset(
        args.datasets, args.data_base_dir, layer_names
    )
    
    # Create train/test split
    print(f"\nCreating train/test split (test_size={args.test_size})...")
    
    # Use the first layer to determine indices for splitting
    first_layer = next(iter(activations_dict.keys()))
    n_samples = len(activations_dict[first_layer])
    indices = np.arange(n_samples)
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=labels
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Train probes for each layer
    all_results = []
    
    for layer_name in layer_names:
        if layer_name not in activations_dict:
            print(f"Skipping {layer_name} (not found)")
            continue
        
        # Get train/test data for this layer
        X = activations_dict[layer_name]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        
        # Train probes
        layer_results = train_and_evaluate_probes(
            X_train, y_train, X_test, y_test,
            args.probe_types, layer_name, output_dir, args.num_epochs
        )
        
        # Add to results
        for probe_type, metrics in layer_results.items():
            result_row = {
                'layer': layer_name,
                'probe_type': probe_type,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                **metrics
            }
            all_results.append(result_row)
    
    # Train probes on averaged activations if requested
    if args.include_averaged and len(layer_names) > 1:
        print("\n--- Training probes on averaged activations ---")
        
        # Compute averaged activations
        averaged_activations = compute_averaged_activations(activations_dict, layer_names)
        
        X_train_avg = averaged_activations[train_indices]
        X_test_avg = averaged_activations[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        
        # Train probes
        avg_results = train_and_evaluate_probes(
            X_train_avg, y_train, X_test_avg, y_test,
            args.probe_types, 'averaged_layers', output_dir, args.num_epochs
        )
        
        # Add to results
        for probe_type, metrics in avg_results.items():
            result_row = {
                'layer': 'averaged_layers',
                'probe_type': probe_type,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                **metrics
            }
            all_results.append(result_row)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(output_dir, 'probe_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")
        
        # Display summary
        print("\n--- Results Summary ---")
        summary_metrics = ['accuracy', 'f1_score']
        if 'auroc' in results_df.columns:
            summary_metrics.append('auroc')
        print(results_df.pivot_table(
            index='layer', 
            columns='probe_type', 
            values=summary_metrics,
            aggfunc='mean'
        ))
        
        # Create plots
        plot_results(results_df, output_dir, args.datasets, args.model_name)
        
    else:
        print("No results to save.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
