#!/usr/bin/env python3
"""Script for extracting activations from datasets.

Extracts activations and saves:
- Dataset with text/responses → CSV file
- Mean activations per layer → separate pickle files (mean across sequence length)

Usage:
    python extract_activations.py --model meta-llama/Llama-3.2-3B-Instruct \
                                   --data data.jsonl \
                                   --output-dir data/activations/my_experiment
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from ign_inf_unlearning.models.activations import (
    _cleanup_gpu_memory,
    get_batch_res_activations,
    get_batch_res_activations_with_generation,
    get_model,
)


# ============================================================================
# Data Loading Functions (Adapt these to your dataset format)
# ============================================================================


def load_jsonl_data(file_path: str, text_field: str = "text") -> List[str]:
    """Load text data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        text_field: Name of field containing text (default: "text")
        
    Returns:
        List of text strings
    """
    sequences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sequences.append(data[text_field])
    return sequences


def load_data(file_path: str, text_field: str = "text") -> List[str]:
    """Load data from JSONL or CSV file.
    
    Args:
        file_path: Path to data file
        text_field: Name of field/column containing text
        
    Returns:
        List of text strings
    """
    file_path = Path(file_path)
    
    if file_path.suffix == ".jsonl":
        return load_jsonl_data(str(file_path), text_field)
    elif file_path.suffix == ".csv":
        sequences = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sequences.append(row[text_field])
        return sequences
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}. Use .jsonl or .csv")


def load_mcqa_data(file_path: str, text_field: str = "question") -> List[Dict[str, Any]]:
    """Load multiple choice question data from CSV file.
    
    Args:
        file_path: Path to CSV file
        text_field: Name of field/column containing question text
        
    Returns:
        List of dictionaries containing question data
    """
    file_path = Path(file_path)
    
    if file_path.suffix != ".csv":
        raise ValueError(f"MCQA data must be CSV format, got: {file_path.suffix}")
    
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different CSV formats
            if text_field in row:
                question_text = row[text_field]
            elif "rewritten" in row:  # real_words_sciency format
                question_text = row["rewritten"]
            else:
                raise ValueError(f"Could not find question text field '{text_field}' in CSV")
            
            # Get original question if available (for real_words_sciency format)
            original_question = row.get("original", None)
            
            # Extract choices/options
            choices = []
            if "choices" in row:  # wmdp_bio_robust format
                choices_str = row["choices"]
                # Parse the string representation of list
                import ast
                import re
                try:
                    # Fix the format: replace newlines between quotes with commas
                    # This handles the case where list elements are separated by newlines instead of commas
                    fixed_str = re.sub(r"'\s*\n\s*'", "', '", choices_str)
                    choices = ast.literal_eval(fixed_str)
                except (ValueError, SyntaxError):
                    choices = [choices_str]  # fallback
            elif "options" in row:  # real_words_sciency format
                choices_str = row["options"]
                import ast
                import re
                try:
                    # Fix the format: replace newlines between quotes with commas
                    # This handles the case where list elements are separated by newlines instead of commas
                    fixed_str = re.sub(r"'\s*\n\s*'", "', '", choices_str)
                    choices = ast.literal_eval(fixed_str)
                except (ValueError, SyntaxError):
                    choices = [choices_str]  # fallback
            
            # Get correct answer
            correct_answer = None
            if "answer" in row:
                try:
                    correct_answer = int(row["answer"])
                except (ValueError, TypeError):
                    correct_answer = row["answer"]
            elif "correct" in row:
                try:
                    correct_answer = int(row["correct"])
                except (ValueError, TypeError):
                    correct_answer = row["correct"]
            
            questions.append({
                "question": question_text,
                "original_question": original_question,
                "choices": choices,
                "correct_answer": correct_answer,
                "original_row": row
            })
    
    return questions


def format_mcqa_prompt(question: str, choices: List[str]) -> str:
    """Format a multiple choice question as a prompt.
    
    Args:
        question: The question text
        choices: List of answer choices
        
    Returns:
        Formatted prompt string
    """
    if not choices:
        return question
    
    prompt = question + "\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"  # A, B, C, D...
    
    prompt += "\nAnswer:"
    return prompt


# ============================================================================
# Saving Functions
# ============================================================================


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    model_name: str = None,
) -> None:
    """Save dataset as CSV and mean activations as per-layer pickle files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save outputs
        model_name: Model name to include in file paths (optional)
    """
    output_dir = Path(output_dir)
    
    # Create model-specific subdirectory if model_name is provided
    if model_name:
        # Clean model name for use in directory name (replace / with _)
        clean_model_name = model_name.replace("/", "_")
        model_output_dir = output_dir / clean_model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = "dataset.csv"
    else:
        model_output_dir = output_dir
        model_output_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = "dataset.csv"
    
    # Save dataset as CSV
    csv_path = model_output_dir / csv_filename
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Determine fieldnames based on what's in the results
        fieldnames = ["text"]
        if results:
            if "original_question" in results[0] and results[0]["original_question"] is not None:
                fieldnames.append("original_question")
            if "generated_text" in results[0]:
                fieldnames.append("generated_text")
            if "model_answer" in results[0]:
                fieldnames.append("model_answer")
            if "correct_answer" in results[0]:
                fieldnames.append("correct_answer")
            if "choices" in results[0]:
                fieldnames.append("choices")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {"text": result.get("text", "")}
            if "original_question" in result and result["original_question"] is not None:
                row["original_question"] = result["original_question"]
            if "generated_text" in result:
                row["generated_text"] = result["generated_text"]
            if "model_answer" in result:
                row["model_answer"] = result["model_answer"]
            if "correct_answer" in result:
                row["correct_answer"] = result["correct_answer"]
            if "choices" in result:
                row["choices"] = str(result["choices"])  # Convert list to string for CSV
            writer.writerow(row)
    
    print(f"Saved dataset ({len(results)} examples) to {csv_path}")
    
    # Collect mean activations by layer (already computed during extraction)
    layer_data = {}
    
    for result in results:
        if "activations" in result:
            for layer_idx, mean_activation in result["activations"].items():
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = []
                
                # Debug: print shape of first activation for each layer
                if len(layer_data[layer_idx]) == 0:  # First activation for this layer
                    print(f"Layer {layer_idx} mean activation shape: {mean_activation.shape}")
                
                layer_data[layer_idx].append(mean_activation)
    
    # Save each layer as separate pickle file with mean activations
    for layer_idx, mean_activations in layer_data.items():
        pkl_filename = f"layer_{layer_idx}.pkl"
        pkl_path = model_output_dir / pkl_filename
        with open(pkl_path, "wb") as f:
            pickle.dump(mean_activations, f)
        print(f"Saved layer {layer_idx}: {len(mean_activations)} mean activations to {pkl_path}")


# ============================================================================
# Main Extraction Function
# ============================================================================


def extract_mcqa_activations_from_file(
    model_name: str,
    data_path: str,
    output_dir: str,
    layers_str: str = "auto",
    batch_size: int = 32,
    max_length: int = 2048,
    text_field: str = "question",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Extract activations from multiple choice questions.
    
    This function:
    1. Loads MCQA data from CSV
    2. Formats questions as prompts with choices
    3. Extracts activations ONLY from the prompt (not generated answer)
    4. Generates model's answer to each question
    5. Saves questions, model answers, correct answers, and activations
    
    Args:
        model_name: HuggingFace model identifier
        data_path: Path to MCQA CSV file
        output_dir: Directory to save outputs
        layers_str: Layer specification ("auto", "all", "0,5,10", etc.)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        text_field: Name of field/column containing question text
        max_new_tokens: Maximum tokens to generate for answer
        temperature: Sampling temperature for answer generation
        verbose: Whether to print detailed progress
        
    Returns:
        List of result dictionaries
    """
    print("=" * 80)
    print("MCQA Activation Extraction")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layers_str}")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")
    
    # Load model
    model, tokenizer = get_model(model_name)
    
    # Load MCQA data
    print(f"Loading MCQA data from {data_path}...")
    questions_data = load_mcqa_data(data_path, text_field=text_field)
    print(f"Loaded {len(questions_data)} questions\n")
    
    # Process in batches
    all_results = []
    num_batches = (len(questions_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(questions_data), batch_size), desc="Processing batches", total=num_batches):
        batch_data = questions_data[i:i+batch_size]
        
        try:
            # Format prompts for this batch
            prompts = []
            for q_data in batch_data:
                prompt = format_mcqa_prompt(q_data["question"], q_data["choices"])
                prompts.append(prompt)
            
            # Extract activations from prompts only (no generation during activation extraction)
            activations, _, input_length = get_batch_res_activations(
                model=model,
                tokenizer=tokenizer,
                outputs=prompts,
                layers_str=layers_str,
                max_length=max_length,
                verbose=verbose,
            )
            
            # Generate answers separately
            answers = []
            for prompt in prompts:
                try:
                    # Generate answer
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                    if hasattr(model, 'device'):
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=temperature > 0,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    
                    # Decode only the generated part
                    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    answers.append(answer)
                    
                except Exception as e:
                    print(f"Error generating answer: {e}")
                    answers.append("")
            
            # Store results
            for j, q_data in enumerate(batch_data):
                # Compute mean activations for each layer
                mean_activations = {}
                for layer_idx in activations.keys():
                    activation = activations[layer_idx][j]
                    
                    # Debug: print shape information
                    if j == 0:  # Only print for first sample to avoid spam
                        print(f"Layer {layer_idx} activation shape: {activation.shape}")
                    
                    if hasattr(activation, 'mean'):
                        # If it's a tensor, compute mean across sequence dimension
                        if len(activation.shape) == 3:  # [batch, seq_len, hidden_size]
                            mean_activation = activation.mean(dim=1)  # Shape: [hidden_size]
                        elif len(activation.shape) == 2:  # [seq_len, hidden_size]
                            mean_activation = activation.mean(dim=0)  # Shape: [hidden_size]
                        else:
                            # Fallback: assume last dimension is sequence length
                            mean_activation = activation.mean(dim=-2)  # Shape: [hidden_size]
                    else:
                        # If it's a numpy array, compute mean across sequence dimension
                        import numpy as np
                        if len(activation.shape) == 3:  # [batch, seq_len, hidden_size]
                            mean_activation = np.mean(activation, axis=1)  # Shape: [hidden_size]
                        elif len(activation.shape) == 2:  # [seq_len, hidden_size]
                            mean_activation = np.mean(activation, axis=0)  # Shape: [hidden_size]
                        else:
                            # Fallback: assume second-to-last dimension is sequence length
                            mean_activation = np.mean(activation, axis=-2)  # Shape: [hidden_size]
                    
                    mean_activations[layer_idx] = mean_activation
                
                result = {
                    "text": q_data["question"],  # Store the question used for activations (rewritten)
                    "model_answer": answers[j] if j < len(answers) else "",
                    "correct_answer": q_data["correct_answer"],
                    "choices": q_data["choices"],
                    "input_length": input_length,
                    "activations": mean_activations
                }
                
                # Add original question if available
                if q_data["original_question"] is not None:
                    result["original_question"] = q_data["original_question"]
                
                all_results.append(result)
        
        except Exception as e:
            print(f"\nError processing batch {i//batch_size}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
        
        finally:
            # Cleanup after each batch
            if 'activations' in locals():
                del activations
            _cleanup_gpu_memory()
    
    print(f"\nProcessed {len(all_results)} questions successfully")
    
    # Save results (CSV + per-layer pickles)
    print("\nSaving results...")
    save_results(all_results, output_dir, model_name)
    
    return all_results


def extract_activations_from_file(
    model_name: str,
    data_path: str,
    output_dir: str,
    layers_str: str = "auto",
    batch_size: int = 32,
    max_length: int = 2048,
    text_field: str = "text",
    with_generation: bool = False,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    verbose: bool = False,
    mcqa_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Extract activations from a dataset file.
    
    Saves:
    - output_dir/dataset.csv (text and optional generated responses)
    - output_dir/layer_{N}.pkl (activations for each layer)
    
    Args:
        model_name: HuggingFace model identifier
        data_path: Path to input data file (.jsonl or .csv)
        output_dir: Directory to save outputs
        layers_str: Layer specification ("auto", "all", "0,5,10", etc.)
                   Default "auto" = every third layer
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        text_field: Name of field/column containing text
        with_generation: Whether to generate new tokens
        max_new_tokens: Maximum tokens to generate (if with_generation=True)
        temperature: Sampling temperature (if with_generation=True)
        verbose: Whether to print detailed progress
        mcqa_mode: Whether to treat data as multiple choice questions
        
    Returns:
        List of result dictionaries
    """
    # Route to MCQA function if in MCQA mode
    if mcqa_mode:
        return extract_mcqa_activations_from_file(
            model_name=model_name,
            data_path=data_path,
            output_dir=output_dir,
            layers_str=layers_str,
            batch_size=batch_size,
            max_length=max_length,
            text_field=text_field,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
    
    print("=" * 80)
    print("Activation Extraction")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layers_str}")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")
    
    # Load model
    model, tokenizer = get_model(model_name)
    
    # Load data
    print(f"Loading data from {data_path}...")
    sequences = load_data(data_path, text_field=text_field)
    print(f"Loaded {len(sequences)} sequences\n")
    
    # Process in batches
    all_results = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches", total=num_batches):
        batch = sequences[i:i+batch_size]
        
        try:
            # Extract activations
            if with_generation:
                activations, outputs, input_length = get_batch_res_activations_with_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch,
                    layers_str=layers_str,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verbose=verbose,
                )
            else:
                activations, outputs, input_length = get_batch_res_activations(
                    model=model,
                    tokenizer=tokenizer,
                    outputs=batch,
                    layers_str=layers_str,
                    max_length=max_length,
                    verbose=verbose,
                )
            
            # Store results
            for j, seq in enumerate(batch):
                # Compute mean activations for each layer
                mean_activations = {}
                for layer_idx in activations.keys():
                    activation = activations[layer_idx][j]
                    
                    # Debug: print shape information
                    if j == 0:  # Only print for first sample to avoid spam
                        print(f"Layer {layer_idx} activation shape: {activation.shape}")
                    
                    if hasattr(activation, 'mean'):
                        # If it's a tensor, compute mean across sequence dimension
                        if len(activation.shape) == 3:  # [batch, seq_len, hidden_size]
                            mean_activation = activation.mean(dim=1)  # Shape: [hidden_size]
                        elif len(activation.shape) == 2:  # [seq_len, hidden_size]
                            mean_activation = activation.mean(dim=0)  # Shape: [hidden_size]
                        else:
                            # Fallback: assume last dimension is sequence length
                            mean_activation = activation.mean(dim=-2)  # Shape: [hidden_size]
                    else:
                        # If it's a numpy array, compute mean across sequence dimension
                        import numpy as np
                        if len(activation.shape) == 3:  # [batch, seq_len, hidden_size]
                            mean_activation = np.mean(activation, axis=1)  # Shape: [hidden_size]
                        elif len(activation.shape) == 2:  # [seq_len, hidden_size]
                            mean_activation = np.mean(activation, axis=0)  # Shape: [hidden_size]
                        else:
                            # Fallback: assume second-to-last dimension is sequence length
                            mean_activation = np.mean(activation, axis=-2)  # Shape: [hidden_size]
                    
                    mean_activations[layer_idx] = mean_activation
                
                result = {
                    "text": seq,
                    "input_length": input_length,
                    "activations": mean_activations
                }
                
                if with_generation and j < len(outputs):
                    result["generated_text"] = outputs[j]
                
                all_results.append(result)
        
        except Exception as e:
            print(f"\nError processing batch {i//batch_size}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
        
        finally:
            # Cleanup after each batch
            del activations
            _cleanup_gpu_memory()
    
    print(f"\nProcessed {len(all_results)} examples successfully")
    
    # Save results (CSV + per-layer pickles)
    print("\nSaving results...")
    save_results(all_results, output_dir, model_name)
    
    return all_results


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (every 3rd layer by default)
  python extract_activations.py --model gpt2 \\
                                 --data data.jsonl \\
                                 --output-dir data/activations/exp1

  # Specify specific layers
  python extract_activations.py --model gpt2 \\
                                 --data data.jsonl \\
                                 --output-dir data/activations/exp2 \\
                                 --layers 0,5,10

  # Extract with text generation
  python extract_activations.py --model gpt2 \\
                                 --data prompts.jsonl \\
                                 --output-dir data/activations/exp3 \\
                                 --with-generation --max-new-tokens 50

  # Extract from multiple choice questions (activations from prompts, record answers)
  python extract_activations.py --model microsoft/phi-2 \\
                                 --data wmdp_bio_robust.csv \\
                                 --output-dir data/activations/mcqa \\
                                 --text-field question --mcqa

Output:
  output-dir/
    └── model-name/               # Model-specific subdirectory
        ├── dataset.csv           # Text sequences (and responses/answers if applicable)
        ├── layer_0.pkl           # Mean activations for layer 0 (list of mean tensors)
        ├── layer_3.pkl           # Mean activations for layer 3
        └── layer_6.pkl           # etc.

MCQA Mode:
  When --mcqa is used, the script:
  - Loads multiple choice questions from CSV
  - Formats questions with choices as prompts
  - Extracts activations ONLY from the prompt (not generated text)
  - Generates model's answer to each question
  - Saves: question text, original question (if available), model answer, correct answer, choices, and mean activations
  - For real_words_sciency format: saves both original and rewritten questions
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to input data file (.jsonl or .csv)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save outputs")
    
    parser.add_argument("--layers", type=str, default="auto",
                        help="Layers to extract (default: auto = every 3rd layer)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for processing (default: 32)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Field/column name containing text (default: 'text')")
    
    parser.add_argument("--with-generation", action="store_true",
                        help="Generate new tokens and extract activations")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum new tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature, 0 = greedy (default: 0.0)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    parser.add_argument("--mcqa", action="store_true",
                        help="Treat data as multiple choice questions (extract activations from prompts, generate answers)")
    
    args = parser.parse_args()

    print("get activations")
    
    # Run extraction
    extract_activations_from_file(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        layers_str=args.layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_field=args.text_field,
        with_generation=args.with_generation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
        mcqa_mode=args.mcqa,
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

