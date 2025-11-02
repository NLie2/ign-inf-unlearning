#!/usr/bin/env python3
"""Compare probe performances between different models.

This script loads probe_results.csv files from different model directories and creates
comparison plots showing one line per model, with separate plots for each probe type.

Usage:
    python scripts/probes/compare_probe_results.py --base_dir outputs/probes \
                                                   --dataset_pattern "*gibberish*" \
                                                   --output_dir outputs/comparisons
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_probe_results(
    base_dir: str, 
    dataset_pattern: Optional[str] = None,
    model_pattern: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """Find all probe_results.csv files in the directory structure.
    
    Args:
        base_dir: Base directory containing model subdirectories
        dataset_pattern: Optional glob pattern to filter dataset directories
        model_pattern: Optional glob pattern to filter model directories
        
    Returns:
        List of tuples (model_name, dataset_name, csv_path)
    """
    base_path = Path(base_dir)
    results = []
    
    # Look for model directories
    model_dirs = []
    if model_pattern:
        model_dirs = list(base_path.glob(model_pattern))
    else:
        model_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Look for dataset directories within each model
        dataset_dirs = []
        if dataset_pattern:
            dataset_dirs = list(model_dir.glob(dataset_pattern))
        else:
            dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            csv_path = dataset_dir / "probe_results.csv"
            
            if csv_path.exists():
                results.append((model_name, dataset_name, str(csv_path)))
                print(f"Found: {model_name} / {dataset_name}")
    
    return results


def load_and_combine_results(
    result_files: List[Tuple[str, str, str]]
) -> pd.DataFrame:
    """Load and combine probe results from multiple files.
    
    Args:
        result_files: List of tuples (model_name, dataset_name, csv_path)
        
    Returns:
        Combined DataFrame with model_name and dataset_name columns added
    """
    all_dfs = []
    
    for model_name, dataset_name, csv_path in result_files:
        try:
            df = pd.read_csv(csv_path)
            df['model_name'] = model_name
            df['dataset_name'] = dataset_name
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {model_name}/{dataset_name}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    
    if not all_dfs:
        raise ValueError("No valid probe results files found")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} total rows")
    print(f"Models: {sorted(combined_df['model_name'].unique())}")
    print(f"Datasets: {sorted(combined_df['dataset_name'].unique())}")
    print(f"Probe types: {sorted(combined_df['probe_type'].unique())}")
    
    return combined_df


def extract_layer_number(layer_name: str) -> int:
    """Extract layer number for sorting."""
    if layer_name == 'averaged_layers':
        return 999  # Put averaged at the end
    try:
        return int(layer_name.split('_')[1])
    except (IndexError, ValueError):
        return 0


def create_comparison_plots(
    df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = ['accuracy', 'f1_score', 'auroc'],
    figsize: Tuple[int, int] = (15, 10),
    dataset_name: Optional[str] = None,
    dataset_pattern: Optional[str] = None,
    model_pattern: Optional[str] = None
) -> None:
    """Create comparison plots showing one line per model.
    
    Args:
        df: Combined DataFrame with results
        output_dir: Directory to save plots
        metrics: List of metrics to plot
        figsize: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique combinations for organizing plots
    datasets = sorted(df['dataset_name'].unique())
    probe_types = sorted(df['probe_type'].unique())
    models = sorted(df['model_name'].unique())
    
    # Set up color palette for models
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    # Create consolidated plots (all models on same graph)
    create_consolidated_plots(df, output_dir, metrics, figsize, model_colors, dataset_name, dataset_pattern, model_pattern)
    
    # Create plots for each dataset and metric combination (original behavior)
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset].copy()
        
        # Add layer numbers for sorting
        dataset_df['layer_num'] = dataset_df['layer'].apply(extract_layer_number)
        dataset_df = dataset_df.sort_values(['layer_num', 'model_name', 'probe_type'])
        
        for metric in metrics:
            if metric not in dataset_df.columns:
                print(f"Warning: Metric '{metric}' not found in data")
                continue
            
            # Create subplot for each probe type
            fig, axes = plt.subplots(1, len(probe_types), figsize=figsize, sharey=True)
            if len(probe_types) == 1:
                axes = [axes]
            
            # Store handles and labels for shared legend
            legend_handles = []
            legend_labels = []
            
            for i, probe_type in enumerate(probe_types):
                ax = axes[i]
                probe_df = dataset_df[dataset_df['probe_type'] == probe_type]
                
                # Plot one line per model
                for model in models:
                    model_df = probe_df[probe_df['model_name'] == model]
                    if len(model_df) == 0:
                        continue
                    
                    # Get unique layers and their values
                    layer_values = model_df.groupby('layer')[metric].mean().reset_index()
                    layer_values['layer_num'] = layer_values['layer'].apply(extract_layer_number)
                    layer_values = layer_values.sort_values('layer_num')
                    
                    # Create x-axis positions
                    x_positions = range(len(layer_values))
                    
                    # Plot line
                    line = ax.plot(
                        x_positions,
                        layer_values[metric],
                        marker='o',
                        linewidth=2,
                        markersize=6,
                        label=model.replace('_', '/'),
                        color=model_colors[model]
                    )[0]
                    
                    # Collect legend info from first subplot only
                    if i == 0:
                        legend_handles.append(line)
                        legend_labels.append(model.replace('_', '/'))
                
                # Customize subplot
                ax.set_title(f'{probe_type.capitalize()} Probe', fontsize=14, fontweight='bold')
                ax.set_xlabel('Layer', fontsize=12)
                if i == 0:  # Only label y-axis on leftmost plot
                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                
                # Set x-axis labels
                if len(probe_df) > 0:
                    # Get layer names for x-axis labels
                    unique_layers = sorted(probe_df['layer'].unique(), key=extract_layer_number)
                    ax.set_xticks(range(len(unique_layers)))
                    ax.set_xticklabels([l.replace('layer_', '') for l in unique_layers], rotation=45)
                
                ax.set_ylim(0.5, 1)
                ax.grid(axis='y', alpha=0.3)
                # Remove individual legends - will add shared legend at bottom
            
            # Add overall title
            fig.suptitle(
                f'{metric.replace("_", " ").title()} Comparison: {dataset}',
                fontsize=16,
                fontweight='bold',
                y=0.95  # Move title up slightly to make room for bottom legend
            )
            
            # Add single legend at the bottom
            if legend_handles:
                fig.legend(
                    legend_handles, 
                    legend_labels, 
                    loc='lower center', 
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=len(legend_labels),
                    fontsize=12
                )
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for bottom legend
            
            # Save plot
            safe_dataset = dataset.replace('/', '_').replace(' ', '_')
            if dataset_name:
                plot_filename = f"probe_comparison_{dataset_name}_{safe_dataset}_{metric}.png"
            else:
                plot_filename = f"probe_comparison_{safe_dataset}_{metric}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_path}")
            plt.close()


def create_mean_performance_bar_charts(
    df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = ['accuracy', 'f1_score', 'auroc'],
    figsize: Tuple[int, int] = (10, 6),
    dataset_name: Optional[str] = None
) -> None:
    """Create separate bar charts for each probe type showing mean performance across all layers.
    
    Args:
        df: Combined DataFrame with results
        output_dir: Directory to save plots
        metrics: List of metrics to plot
        figsize: Figure size
    """
    probe_types = sorted(df['probe_type'].unique())
    models = sorted(df['model_name'].unique())
    
    # Exclude averaged_layers from mean calculation
    df_regular = df[df['layer'] != 'averaged_layers'].copy()
    
    for metric in metrics:
        if metric not in df_regular.columns:
            print(f"Warning: Metric '{metric}' not found in data")
            continue
        
        # Calculate mean performance for each model and probe type
        mean_performance = df_regular.groupby(['model_name', 'probe_type'])[metric].mean().reset_index()
        
        # Create separate bar chart for each probe type
        for probe_type in probe_types:
            probe_data = mean_performance[mean_performance['probe_type'] == probe_type]
            
            # Get values for all models
            model_values = []
            for model in models:
                model_row = probe_data[probe_data['model_name'] == model]
                if not model_row.empty:
                    model_values.append(model_row[metric].iloc[0])
                else:
                    model_values.append(0)
            
            # Filter out zero values for y-axis scaling
            non_zero_values = [v for v in model_values if v > 0]
            if not non_zero_values:
                continue
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set up positions for bars (close together)
            x = np.arange(len(models))
            width = 0.6  # Wider bars since we only have one set
            
            # Create bars with different colors for each model
            colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
            bars = ax.bar(
                x, 
                model_values, 
                width, 
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=1
            )
            
            # Add value labels on bars
            for bar, value in zip(bars, model_values):
                if value > 0:  # Only show label if there's actual data
                    height = bar.get_height()
                    ax.annotate(f'{value:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=12,
                               fontweight='bold')
            
            # Set y-axis limits based on data range to make differences more apparent
            min_val = min(non_zero_values)
            max_val = max(non_zero_values)
            range_val = max_val - min_val
            
            # Add some padding but focus on the data range
            y_min = max(0, min_val - range_val * 0.1)
            y_max = min(1, max_val + range_val * 0.1)
            
            # If the range is very small, ensure we have at least some visible range
            if range_val < 0.01:
                y_min = max(0, min_val - 0.005)
                y_max = min(1, max_val + 0.005)
            
            ax.set_ylim(y_min, y_max)
            
            # Customize the plot
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Mean {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_title(f'{probe_type.capitalize()} Probe: Mean {metric.replace("_", " ").title()} Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([model.replace('_', '/') for model in models], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save bar chart
            if dataset_name:
                plot_filename = f"mean_performance_bar_chart_{dataset_name}_{probe_type}_{metric}.png"
            else:
                plot_filename = f"mean_performance_bar_chart_{probe_type}_{metric}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved mean performance bar chart: {plot_path}")
            plt.close()


def create_consolidated_plots(
    df: pd.DataFrame,
    output_dir: str,
    metrics: List[str],
    figsize: Tuple[int, int],
    model_colors: Dict[str, any],
    dataset_name: Optional[str] = None,
    dataset_pattern: Optional[str] = None,
    model_pattern: Optional[str] = None
) -> None:
    """Create consolidated plots with all models on the same graph.
    
    Args:
        df: Combined DataFrame with results
        output_dir: Directory to save plots
        metrics: List of metrics to plot
        figsize: Figure size
        model_colors: Dictionary mapping model names to colors
    """
    probe_types = sorted(df['probe_type'].unique())
    models = sorted(df['model_name'].unique())
    
    # Add layer numbers for sorting
    df_sorted = df.copy()
    df_sorted['layer_num'] = df_sorted['layer'].apply(extract_layer_number)
    df_sorted = df_sorted.sort_values(['layer_num', 'model_name', 'probe_type'])
    
    for metric in metrics:
        if metric not in df_sorted.columns:
            print(f"Warning: Metric '{metric}' not found in data")
            continue
        
        # Create subplot for each probe type with wider individual graphs
        fig, axes = plt.subplots(1, len(probe_types), figsize=(figsize[0] * 1.2, figsize[1]), sharey=True)
        if len(probe_types) == 1:
            axes = [axes]
        
        # Store handles and labels for shared legend
        legend_handles = []
        legend_labels = []
        
        for i, probe_type in enumerate(probe_types):
            ax = axes[i]
            probe_df = df_sorted[df_sorted['probe_type'] == probe_type]
            
            # Find common layers across all models
            all_layers = set()
            model_data = {}
            
            for model in models:
                model_df = probe_df[probe_df['model_name'] == model]
                if len(model_df) == 0:
                    continue
                
                # Group by layer and take mean across datasets for this model
                layer_values = model_df.groupby('layer')[metric].mean().reset_index()
                layer_values['layer_num'] = layer_values['layer'].apply(extract_layer_number)
                layer_values = layer_values.sort_values('layer_num')
                
                model_data[model] = layer_values
                all_layers.update(layer_values['layer'].tolist())
            
            # Sort layers
            common_layers = sorted(all_layers, key=extract_layer_number)
            
            # Plot one line per model
            for model in models:
                if model not in model_data:
                    continue
                
                layer_values = model_data[model]
                
                # Create x-axis positions and y-values for common layers
                x_positions = []
                y_values = []
                
                for j, layer in enumerate(common_layers):
                    layer_row = layer_values[layer_values['layer'] == layer]
                    if not layer_row.empty:
                        x_positions.append(j)
                        y_values.append(layer_row[metric].iloc[0])
                
                if x_positions:  # Only plot if we have data
                    line = ax.plot(
                        x_positions,
                        y_values,
                        marker='o',
                        linewidth=3,
                        markersize=8,
                        label=model.replace('_', '/'),
                        color=model_colors[model]
                    )[0]
                    
                    # Collect legend info from first subplot only
                    if i == 0:
                        legend_handles.append(line)
                        legend_labels.append(model.replace('_', '/'))
            
            # Customize subplot
            ax.set_title(f'{probe_type.capitalize()} Probe', fontsize=14, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=12)
            if i == 0:  # Only label y-axis on leftmost plot
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            
            # Set x-axis labels
            if common_layers:
                ax.set_xticks(range(len(common_layers)))
                ax.set_xticklabels([l.replace('layer_', '') for l in common_layers], rotation=45)
            
            ax.set_ylim(0.5, 1)
            ax.grid(axis='y', alpha=0.3)
            # Remove individual legends
        
        # Add overall title
        fig.suptitle(
            f'{metric.replace("_", " ").title()} Comparison: All Models',
            fontsize=16,
            fontweight='bold',
            y=0.95  # Move title up slightly to make room for bottom legend
        )
        
        # Add single legend at the bottom
        if legend_handles:
            fig.legend(
                legend_handles, 
                legend_labels, 
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(legend_labels),
                fontsize=12
            )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for bottom legend
        
        # Save consolidated plot
        filename_parts = ["consolidated_probe_comparison"]
        
        if dataset_pattern:
            # Clean up pattern for filename
            clean_pattern = dataset_pattern.replace('*', '').replace('/', '_')
            filename_parts.append(clean_pattern)
        
        if model_pattern:
            # Clean up pattern for filename
            clean_pattern = model_pattern.replace('*', '').replace('/', '_')
            filename_parts.append(clean_pattern)
        
        if dataset_name:
            filename_parts.append(dataset_name)
        
        filename_parts.append(metric)
        plot_filename = "_".join(filename_parts) + ".png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved consolidated plot: {plot_path}")
        plt.close()


def create_performance_difference_plots(
    df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = ['accuracy', 'f1_score', 'auroc'],
    figsize: Tuple[int, int] = (12, 8),
    dataset_pattern: Optional[str] = None,
    model_pattern: Optional[str] = None
) -> None:
    """Create plots showing performance differences between models per layer.
    
    Args:
        df: Combined DataFrame with results
        output_dir: Directory to save plots
        metrics: List of metrics to plot
        figsize: Figure size
        dataset_pattern: Dataset pattern for filename
        model_pattern: Model pattern for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique combinations
    datasets = sorted(df['dataset_name'].unique())
    probe_types = sorted(df['probe_type'].unique())
    models = sorted(df['model_name'].unique())
    
    if len(models) != 2:
        print(f"Warning: Performance difference plots require exactly 2 models, found {len(models)}")
        return
    
    model1, model2 = models[0], models[1]
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data")
            continue
        
        # Create subplot for each probe type
        fig, axes = plt.subplots(1, len(probe_types), figsize=figsize, sharey=True)
        if len(probe_types) == 1:
            axes = [axes]
        
        for i, probe_type in enumerate(probe_types):
            ax = axes[i]
            probe_df = df[df['probe_type'] == probe_type].copy()
            
            # Add layer numbers for sorting
            probe_df['layer_num'] = probe_df['layer'].apply(extract_layer_number)
            probe_df = probe_df.sort_values(['layer_num', 'model_name'])
            
            # Calculate differences for each dataset
            differences = []
            layers = []
            colors = []
            
            for dataset in datasets:
                dataset_df = probe_df[probe_df['dataset_name'] == dataset]
                
                # Get performance for each model
                model1_data = dataset_df[dataset_df['model_name'] == model1]
                model2_data = dataset_df[dataset_df['model_name'] == model2]
                
                if model1_data.empty or model2_data.empty:
                    continue
                
                # Calculate mean performance per layer
                model1_layers = model1_data.groupby('layer')[metric].mean()
                model2_layers = model2_data.groupby('layer')[metric].mean()
                
                # Find common layers
                common_layers = set(model1_layers.index) & set(model2_layers.index)
                common_layers = sorted(common_layers, key=extract_layer_number)
                
                for layer in common_layers:
                    if layer == 'averaged_layers':
                        continue  # Skip averaged layers for this comparison
                    
                    diff = model1_layers[layer] - model2_layers[layer]
                    differences.append(diff)
                    layers.append(f"{layer.replace('layer_', '')}")
                    
                    # Color coding: red if model1 better, blue if model2 better, gray if not different
                    if abs(diff) < 0.01:
                        colors.append('gray')
                    elif diff > 0:
                        colors.append('red')
                    else:
                        colors.append('blue')
            
            # Create bar plot
            x_positions = range(len(differences))
            bars = ax.bar(x_positions, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Customize plot
            ax.set_title(f'{probe_type.capitalize()} Probe: {model1.replace("_", "/")} vs {model2.replace("_", "/")}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=12)
            if i == 0:
                ax.set_ylabel(f'{metric.replace("_", " ").title()} Difference\n({model1.replace("_", "/")} - {model2.replace("_", "/")})', fontsize=12)
            
            # Set x-axis labels
            if layers:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(layers, rotation=45)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label=f'{model1.replace("_", "/")} better'),
                Patch(facecolor='blue', alpha=0.7, label=f'{model2.replace("_", "/")} better'),
                Patch(facecolor='gray', alpha=0.7, label='Not different (< 0.01)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, differences):
                height = bar.get_height()
                ax.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -3),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=10,
                           fontweight='bold')
        
        # Add overall title
        fig.suptitle(
            f'{metric.replace("_", " ").title()} Performance Differences by Layer',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save plot
        filename_parts = ["performance_difference"]
        
        if dataset_pattern:
            clean_pattern = dataset_pattern.replace('*', '').replace('/', '_')
            filename_parts.append(clean_pattern)
        
        if model_pattern:
            clean_pattern = model_pattern.replace('*', '').replace('/', '_')
            filename_parts.append(clean_pattern)
        
        filename_parts.append(metric)
        plot_filename = "_".join(filename_parts) + ".png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance difference plot: {plot_path}")
        plt.close()


def create_summary_table(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """Create summary tables comparing models.
    
    Args:
        df: Combined DataFrame with results
        output_dir: Directory to save tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary by model and probe type
    summary_stats = []
    
    for dataset in df['dataset_name'].unique():
        dataset_df = df[df['dataset_name'] == dataset]
        
        for model in dataset_df['model_name'].unique():
            model_df = dataset_df[dataset_df['model_name'] == model]
            
            for probe_type in model_df['probe_type'].unique():
                probe_df = model_df[model_df['probe_type'] == probe_type]
                
                # Exclude averaged_layers for statistics
                regular_layers = probe_df[probe_df['layer'] != 'averaged_layers']
                avg_layers = probe_df[probe_df['layer'] == 'averaged_layers']
                
                for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auroc']:
                    if metric in probe_df.columns:
                        stats = {
                            'dataset': dataset,
                            'model': model,
                            'probe_type': probe_type,
                            'metric': metric,
                            'min': regular_layers[metric].min() if not regular_layers.empty else np.nan,
                            'max': regular_layers[metric].max() if not regular_layers.empty else np.nan,
                            'mean': regular_layers[metric].mean() if not regular_layers.empty else np.nan,
                            'std': regular_layers[metric].std() if not regular_layers.empty else np.nan,
                            'best_layer': regular_layers.loc[regular_layers[metric].idxmax(), 'layer'] if not regular_layers.empty else np.nan,
                            'averaged_layers': avg_layers[metric].iloc[0] if not avg_layers.empty else np.nan
                        }
                        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary table
    summary_path = os.path.join(output_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    
    # Create pivot tables for easier comparison
    for metric in ['accuracy', 'f1_score', 'auroc']:
        if metric in summary_df['metric'].values:
            metric_df = summary_df[summary_df['metric'] == metric]
            
            # Pivot by model and probe type (mean values)
            pivot_mean = metric_df.pivot_table(
                index=['dataset', 'model'],
                columns='probe_type',
                values='mean',
                aggfunc='first'
            )
            
            pivot_path = os.path.join(output_dir, f'model_comparison_{metric}_mean.csv')
            pivot_mean.to_csv(pivot_path)
            print(f"Pivot table saved to: {pivot_path}")


def main():
    """Main function to compare probe results across models."""
    parser = argparse.ArgumentParser(
        description="Compare probe performances between different models."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="outputs/probes",
        help="Base directory containing model subdirectories with probe results"
    )
    parser.add_argument(
        "--dataset_pattern",
        type=str,
        help="Glob pattern to filter dataset directories (e.g., '*gibberish*')"
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        help="Glob pattern to filter model directories (e.g., '*phi*')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/probe_comparisons",
        help="Directory to save comparison plots and tables"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=['accuracy', 'f1_score', 'auroc'],
        choices=['accuracy', 'f1_score', 'precision', 'recall', 'auroc'],
        help="Metrics to plot"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[15, 10],
        help="Figure size as width height"
    )
    
    args = parser.parse_args()
    
    # Validate base directory
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        sys.exit(1)
    
    print(f"Searching for probe results in: {args.base_dir}")
    if args.dataset_pattern:
        print(f"Dataset pattern: {args.dataset_pattern}")
    if args.model_pattern:
        print(f"Model pattern: {args.model_pattern}")
    
    # Find all probe result files
    result_files = find_probe_results(
        args.base_dir,
        args.dataset_pattern,
        args.model_pattern
    )
    
    if not result_files:
        print("No probe result files found!")
        sys.exit(1)
    
    print(f"\nFound {len(result_files)} probe result files")
    
    # Load and combine results
    combined_df = load_and_combine_results(result_files)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(
        combined_df,
        args.output_dir,
        args.metrics,
        tuple(args.figsize),
        dataset_pattern=args.dataset_pattern,
        model_pattern=args.model_pattern
    )
    
    # Create mean performance bar charts
    print("\nCreating mean performance bar charts...")
    create_mean_performance_bar_charts(
        combined_df,
        args.output_dir,
        args.metrics,
        tuple(args.figsize)
    )
    
    # Create performance difference plots
    print("\nCreating performance difference plots...")
    create_performance_difference_plots(
        combined_df,
        args.output_dir,
        args.metrics,
        tuple(args.figsize),
        args.dataset_pattern,
        args.model_pattern
    )
    
    # Create summary tables
    print("\nCreating summary tables...")
    create_summary_table(combined_df, args.output_dir)
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, 'combined_probe_results.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved to: {combined_path}")
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
