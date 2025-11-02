"""Visualization utilities for probe analysis and model interpretability."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_probe_results(
    results_df: pd.DataFrame,
    metric: str = 'f1_score',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    title_suffix: str = ""
) -> plt.Figure:
    """Plot probe results across layers and probe types.
    
    Args:
        results_df: DataFrame with columns ['layer', 'probe_type', metric]
        metric: Metric to plot ('accuracy', 'f1_score', 'precision', 'recall')
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        title_suffix: Additional text to add to the title
        
    Returns:
        matplotlib Figure object
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create pivot table for plotting
    pivot_df = results_df.pivot(index='layer', columns='probe_type', values=metric)
    
    # Sort layers naturally (handle both numeric and averaged_layers)
    layer_order = sorted(
        pivot_df.index,
        key=lambda x: (
            x == 'averaged_layers',
            int(x.split('_')[-1]) if x != 'averaged_layers' and x.split('_')[-1].isdigit() else float('inf')
        )
    )
    pivot_df = pivot_df.reindex(layer_order)
    
    # Create line plot
    for probe_type in pivot_df.columns:
        ax.plot(
            range(len(pivot_df.index)),
            pivot_df[probe_type],
            marker='o',
            linewidth=3,
            markersize=8,
            label=probe_type.capitalize()
        )
    
    # Customize plot
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_title(f'Probe {metric.replace("_", " ").title()} by Layer{title_suffix}', 
                fontsize=16, fontweight='bold')
    
    # Set x-axis labels
    clean_labels = [
        'Avg' if layer == 'averaged_layers' 
        else layer.split('_')[-1] if layer.startswith('layer_') 
        else layer 
        for layer in pivot_df.index
    ]
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(clean_labels, fontweight='bold')
    
    # Set y-axis limits and formatting
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add legend and grid
    ax.legend(fontsize=12, title='Probe Type', title_fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot a confusion matrix with proper formatting.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: Optional list of class names
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(conf_matrix))]
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_probe_comparison(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a comprehensive comparison plot of probe performance.
    
    Args:
        results_df: DataFrame with probe results
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create pivot table
        pivot_df = results_df.pivot(index='layer', columns='probe_type', values=metric)
        
        # Sort layers
        layer_order = sorted(
            pivot_df.index,
            key=lambda x: (
                x == 'averaged_layers',
                int(x.split('_')[-1]) if x != 'averaged_layers' and x.split('_')[-1].isdigit() else float('inf')
            )
        )
        pivot_df = pivot_df.reindex(layer_order)
        
        # Plot bars
        pivot_df.plot(kind='bar', ax=ax, rot=45, width=0.8)
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Probe Type')
        
        # Clean x-axis labels
        clean_labels = [
            'Avg' if label.get_text() == 'averaged_layers'
            else label.get_text().split('_')[-1] if label.get_text().startswith('layer_')
            else label.get_text()
            for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(clean_labels, rotation=45, ha='right')
    
    plt.suptitle('Probe Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_activation_statistics(
    activations: np.ndarray,
    layer_name: str,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot statistics of activation values for a given layer.
    
    Args:
        activations: Activation array (samples x features)
        layer_name: Name of the layer
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Flatten activations for some plots
    flat_activations = activations.flatten()
    
    # 1. Histogram of activation values
    axes[0, 0].hist(flat_activations, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Activation Values')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Mean activation per feature
    mean_activations = np.mean(activations, axis=0)
    axes[0, 1].plot(mean_activations, alpha=0.7)
    axes[0, 1].set_title('Mean Activation per Feature')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Mean Activation')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Standard deviation per feature
    std_activations = np.std(activations, axis=0)
    axes[1, 0].plot(std_activations, alpha=0.7, color='orange')
    axes[1, 0].set_title('Standard Deviation per Feature')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Correlation matrix (sample if too large)
    if activations.shape[1] > 100:
        # Sample features for correlation plot
        sample_indices = np.random.choice(activations.shape[1], 100, replace=False)
        sample_activations = activations[:, sample_indices]
        title_suffix = ' (100 random features)'
    else:
        sample_activations = activations
        title_suffix = ''
    
    corr_matrix = np.corrcoef(sample_activations.T)
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title(f'Feature Correlation Matrix{title_suffix}')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.suptitle(f'Activation Statistics: {layer_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activation statistics plot saved to: {save_path}")
    
    return fig


def create_probe_summary_table(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Create a summary table of probe results.
    
    Args:
        results_df: DataFrame with probe results
        save_path: Optional path to save the table as CSV
        
    Returns:
        Summary DataFrame
    """
    # Calculate summary statistics
    summary_stats = []
    
    for probe_type in results_df['probe_type'].unique():
        probe_data = results_df[results_df['probe_type'] == probe_type]
        
        # Separate averaged layers from regular layers
        regular_layers = probe_data[probe_data['layer'] != 'averaged_layers']
        avg_layers = probe_data[probe_data['layer'] == 'averaged_layers']
        
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if metric in probe_data.columns:
                stats = {
                    'probe_type': probe_type,
                    'metric': metric,
                    'min': regular_layers[metric].min() if not regular_layers.empty else np.nan,
                    'max': regular_layers[metric].max() if not regular_layers.empty else np.nan,
                    'mean': regular_layers[metric].mean() if not regular_layers.empty else np.nan,
                    'std': regular_layers[metric].std() if not regular_layers.empty else np.nan,
                    'averaged_layers': avg_layers[metric].iloc[0] if not avg_layers.empty else np.nan
                }
                summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"Summary table saved to: {save_path}")
    
    return summary_df


def plot_layer_progression(
    results_df: pd.DataFrame,
    metric: str = 'f1_score',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot how probe performance changes across layers.
    
    Args:
        results_df: DataFrame with probe results
        metric: Metric to plot
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out averaged layers for progression plot
    regular_layers = results_df[results_df['layer'] != 'averaged_layers'].copy()
    
    if regular_layers.empty:
        print("No regular layers found for progression plot")
        return fig
    
    # Extract layer numbers for sorting
    regular_layers['layer_num'] = regular_layers['layer'].str.extract(r'layer_(\d+)').astype(int)
    regular_layers = regular_layers.sort_values('layer_num')
    
    # Plot progression for each probe type
    for probe_type in regular_layers['probe_type'].unique():
        probe_data = regular_layers[regular_layers['probe_type'] == probe_type]
        
        ax.plot(
            probe_data['layer_num'],
            probe_data[metric],
            marker='o',
            linewidth=2,
            markersize=6,
            label=probe_type.capitalize()
        )
    
    ax.set_xlabel('Layer Number', fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} Progression Across Layers', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer progression plot saved to: {save_path}")
    
    return fig
