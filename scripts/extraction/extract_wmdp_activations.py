#!/usr/bin/env python3
"""Extract activations from WMDP-BIO-FORGET MCQA datasets.

This script handles the specific format of WMDP multiple-choice questions
and extracts activations for PROMPT TOKENS ONLY (questions, not answer choices).

Usage:
    python extract_wmdp_activations.py --model gpt2 \
                                        --data-dir data/datasets/input_data/wmdp-bio-forget-mcqa \
                                        --output-dir data/activations/wmdp_exp1
"""

import argparse
import ast
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from ign_inf_unlearning.models.activations import (
    _cleanup_gpu_memory,
    format_prompts_from_strings,
    get_batch_res_activations,
    get_model,
)


def format_mcqa(question: str, choices: List[str], answer_idx: int = None) -> str:
    """Format a multiple-choice question with choices.
    
    Args:
        question: The question text
        choices: List of answer choices
        answer_idx: Index of correct answer (optional, for reference)
        
    Returns:
        Formatted string like:
        "Q: What is...?
         A) Choice 1
         B) Choice 2
         C) Choice 3
         D) Choice 4"
    """
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    formatted = f"Q: {question}\n"
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else str(i)
        formatted += f"{label}) {choice}\n"
    
    return formatted.strip()


def load_wmdp_robust(file_path: str) -> pd.DataFrame:
    """Load wmdp_bio_robust.csv format.
    
    Format: answer, question, choices, config
    
    Note: We only keep the question text for activation extraction (prompt tokens only).
    """
    df = pd.read_csv(file_path)
    
    # Parse choices (they're stored as string representation of list)
    df['choices_list'] = df['choices'].apply(ast.literal_eval)
    
    # Format as MCQA for reference, but use only question for activations
    df['formatted_text'] = df.apply(
        lambda row: format_mcqa(row['question'], row['choices_list'], row['answer']),
        axis=1
    )
    
    # Use only the question text for activation extraction (prompt tokens only)
    df['prompt_text'] = df['question']
    
    # Rename for consistency
    df['correct_answer'] = df['answer']
    df['category'] = df['config']
    
    return df[['prompt_text', 'formatted_text', 'question', 'choices_list', 'correct_answer', 'category']]


def load_rewritten_format(file_path: str, style: str) -> pd.DataFrame:
    """Load gibberish/nonsensical/real_words formats.
    
    Format: custom_id, style, index, original, rewritten, options, correct
    
    Note: We only keep the question text for activation extraction (prompt tokens only).
    """
    df = pd.read_csv(file_path)
    
    # Parse options
    df['choices_list'] = df['options'].apply(ast.literal_eval)
    
    # Use 'rewritten' as the question (the perturbed version)
    df['formatted_text'] = df.apply(
        lambda row: format_mcqa(row['rewritten'], row['choices_list'], row['correct']),
        axis=1
    )
    
    # Use only the question text for activation extraction (prompt tokens only)
    df['prompt_text'] = df['rewritten']
    
    df['correct_answer'] = df['correct']
    df['category'] = style
    df['question'] = df['rewritten']
    
    return df[['prompt_text', 'formatted_text', 'question', 'choices_list', 'correct_answer', 'category']]


def load_all_wmdp_datasets(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all WMDP datasets from directory.
    
    Args:
        data_dir: Directory containing WMDP CSV files
        
    Returns:
        Dictionary mapping dataset name to DataFrame
    """
    data_dir = Path(data_dir)
    datasets = {}
    
    # Load wmdp_bio_robust
    robust_path = data_dir / "wmdp_bio_robust.csv"
    if robust_path.exists():
        print(f"Loading {robust_path.name}...")
        datasets['robust'] = load_wmdp_robust(str(robust_path))
        print(f"  Loaded {len(datasets['robust'])} questions")
    
    # Load rewritten datasets
    rewritten_files = [
        ('gibberish', 'gibberish.csv'),
        ('nonsensical_biology', 'nonsensical_biology.csv'),
        ('real_words_sciency', 'real_words_sciency.csv'),
    ]
    
    for name, filename in rewritten_files:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"Loading {filename}...")
            datasets[name] = load_rewritten_format(str(file_path), name)
            print(f"  Loaded {len(datasets[name])} questions")
    
    return datasets


def extract_wmdp_activations(
    model_name: str,
    data_dir: str,
    output_dir: str,
    layers_str: str = "auto",
    batch_size: int = 32,
    max_length: int = 2048,
    datasets: List[str] = None,
    apply_chat_template: bool = False,
    verbose: bool = False,
):
    """Extract activations from WMDP datasets (PROMPT TOKENS ONLY).
    
    This function extracts activations only from question text (prompts),
    not from answer choices. This ensures we capture the model's internal
    representations when processing the question, before seeing the answers.
    
    Args:
        model_name: HuggingFace model identifier
        data_dir: Directory containing WMDP CSV files
        output_dir: Base directory to save outputs (subdirs per dataset)
        layers_str: Layer specification
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        datasets: List of dataset names to process (None = all)
        apply_chat_template: Whether to apply tokenizer's chat template to prompts
        verbose: Print detailed progress
    """
    print("=" * 80)
    print("WMDP-BIO-FORGET MCQA Activation Extraction (PROMPT TOKENS ONLY)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Layers: {layers_str}")
    print(f"Chat template: {'Applied' if apply_chat_template else 'Not applied (raw text)'}")
    print("Mode: Extract activations from question text only (not answer choices)")
    print("=" * 80 + "\n")
    
    # Load model
    print("Loading model...")
    model, tokenizer = get_model(model_name)
    
    # Load datasets
    print("\nLoading WMDP datasets...")
    all_datasets = load_all_wmdp_datasets(data_dir)
    
    if datasets:
        all_datasets = {k: v for k, v in all_datasets.items() if k in datasets}
    
    if not all_datasets:
        raise ValueError(f"No datasets found in {data_dir}")
    
    print(f"\nProcessing {len(all_datasets)} dataset(s): {list(all_datasets.keys())}\n")
    
    # Process each dataset
    for dataset_name, df in all_datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")
        
        # Create output directory for this dataset
        dataset_output_dir = Path(output_dir) / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get prompt texts (questions only, not answer choices)
        texts = df['prompt_text'].tolist()
        
        # Apply chat template if requested
        if apply_chat_template:
            if verbose:
                print(f"Applying chat template to {len(texts)} prompts...")
            texts = format_prompts_from_strings(tokenizer, texts)
            if verbose and len(texts) > 0:
                print(f"Example formatted prompt:\n{texts[0][:200]}...\n")
        
        # Extract activations in batches
        all_activations = {layer_idx: [] for layer_idx in range(100)}  # Placeholder
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {dataset_name}", total=num_batches):
            batch = texts[i:i+batch_size]
            
            try:
                activations, _, _ = get_batch_res_activations(
                    model=model,
                    tokenizer=tokenizer,
                    outputs=batch,
                    layers_str=layers_str,
                    max_length=max_length,
                    verbose=verbose,
                )
                
                # Collect activations
                for layer_idx, layer_acts in activations.items():
                    if layer_idx not in all_activations:
                        all_activations[layer_idx] = []
                    all_activations[layer_idx].extend(layer_acts)
                
            except Exception as e:
                print(f"\nError processing batch {i//batch_size}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
            
            finally:
                del activations
                _cleanup_gpu_memory()
        
        # Save dataset CSV
        csv_path = dataset_output_dir / "dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved dataset to {csv_path}")
        
        # Save activations per layer
        for layer_idx, layer_acts in all_activations.items():
            if len(layer_acts) > 0:
                pkl_path = dataset_output_dir / f"layer_{layer_idx}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(layer_acts, f)
                print(f"Saved layer {layer_idx}: {len(layer_acts)} activations")
        
        print(f"\nCompleted {dataset_name}: {len(df)} examples")
    
    print("\n✓ All datasets processed!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from WMDP-BIO-FORGET MCQA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from all datasets (raw text)
  python extract_wmdp_activations.py --model gpt2 \\
                                      --data-dir data/datasets/input_data/wmdp-bio-forget-mcqa \\
                                      --output-dir data/activations/wmdp_exp1

  # Extract with chat template (for instruction-tuned models)
  python extract_wmdp_activations.py --model meta-llama/Llama-3.2-3B-Instruct \\
                                      --data-dir data/datasets/input_data/wmdp-bio-forget-mcqa \\
                                      --output-dir data/activations/wmdp_exp2 \\
                                      --apply-chat-template

  # Extract from specific datasets only
  python extract_wmdp_activations.py --model gpt2 \\
                                      --data-dir data/datasets/input_data/wmdp-bio-forget-mcqa \\
                                      --output-dir data/activations/wmdp_exp3 \\
                                      --datasets robust gibberish

Output structure:
  output-dir/
    ├── robust/
    │   ├── dataset.csv
    │   ├── layer_0.pkl
    │   └── layer_3.pkl
    ├── gibberish/
    │   ├── dataset.csv
    │   ├── layer_0.pkl
    │   └── layer_3.pkl
    └── ...
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing WMDP CSV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base directory to save outputs")
    
    parser.add_argument("--layers", type=str, default="auto",
                        help="Layers to extract (default: auto = every 3rd layer)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for processing (default: 32)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    
    parser.add_argument("--datasets", nargs="+", 
                        choices=["robust", "gibberish", "nonsensical_biology", "real_words_sciency"],
                        help="Specific datasets to process (default: all)")
    
    parser.add_argument("--apply-chat-template", action="store_true",
                        help="Apply tokenizer's chat template to prompts (for instruction-tuned models)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    
    args = parser.parse_args()
    
    extract_wmdp_activations(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layers_str=args.layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        datasets=args.datasets,
        apply_chat_template=args.apply_chat_template,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

