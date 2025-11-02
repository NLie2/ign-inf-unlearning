"""Core utilities for extracting activations from transformer models.

This module provides dataset-agnostic functions for:
- Loading models and tokenizers
- Registering activation hooks
- Extracting residual stream activations
- Managing GPU memory
"""

import gc
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Layer and Model Utilities
# ============================================================================


def suggest_layer(total_layers: int) -> str:
    """Suggest layer indices for activation extraction (every third layer).
    
    Args:
        total_layers: Total number of layers in the model
        
    Returns:
        Comma-separated string of suggested layer indices
    """
    layers = list(range(0, total_layers, 3))
    return ",".join(map(str, layers))


def parse_layers_arg(layers_str: str, total_layers: int) -> List[int]:
    """Parse layer specification string into list of layer indices.
    
    Supports various formats:
    - "auto": suggests evenly-spaced layers
    - "all": all layers
    - "0,5,10": specific comma-separated indices
    - "0-10": range (inclusive)
    - "0-10:2": range with step
    
    Args:
        layers_str: String specifying which layers to extract
        total_layers: Total number of layers in the model
        
    Returns:
        List of layer indices to extract
        
    Raises:
        ValueError: If format is invalid or indices are out of range
    """
    layers_str = layers_str.strip().lower()
    
    if layers_str == "auto":
        suggested = suggest_layer(total_layers)
        print(f"Auto-selecting layers: {suggested}")
        layers_str = suggested
    elif layers_str == "all":
        return list(range(total_layers))
    
    # Parse the specification
    layer_indices = []
    parts = layers_str.split(",")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check for range specification
        if "-" in part and not part.startswith("-"):
            range_parts = part.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format: {part}")
            
            start_str, end_str = range_parts
            
            # Check for step specification
            if ":" in end_str:
                end_str, step_str = end_str.split(":")
                step = int(step_str)
            else:
                step = 1
            
            start = int(start_str)
            end = int(end_str)
            
            if start < 0 or end >= total_layers:
                raise ValueError(f"Layer range {start}-{end} out of bounds [0, {total_layers-1}]")
            if start > end:
                raise ValueError(f"Invalid range: start ({start}) > end ({end})")
            
            layer_indices.extend(range(start, end + 1, step))
        else:
            # Single layer index
            idx = int(part)
            if idx < 0 or idx >= total_layers:
                raise ValueError(f"Layer index {idx} out of bounds [0, {total_layers-1}]")
            layer_indices.append(idx)
    
    # Remove duplicates and sort
    layer_indices = sorted(set(layer_indices))
    
    if not layer_indices:
        raise ValueError("No valid layer indices specified")
    
    return layer_indices


def get_pad_token(model_id: str, tokenizer: AutoTokenizer) -> None:
    """Set padding token for tokenizer if not already set.
    
    Args:
        model_id: Model identifier (used for special handling)
        tokenizer: Tokenizer to configure
    """
    if tokenizer.pad_token is None:
        if "llama" in model_id.lower():
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to: {tokenizer.pad_token}")


def get_model(model_name: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and its tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ("auto", "cuda", "cpu")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    get_pad_token(model_name, tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    model.eval()
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def get_res_layers_to_enumerate(model: AutoModelForCausalLM) -> List:
    """Get list of residual stream layers from the model.
    
    Supports various model architectures (Llama, GPT, GPT-NeoX, Pythia, etc.)
    
    Args:
        model: The transformer model
        
    Returns:
        List of layer modules
    """
    # Common direct matches
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama-style models
        return list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-style models
        return list(model.transformer.h)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return list(model.transformer.layers)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # Encoder models
        return list(model.encoder.layer)
    
    # EleutherAI GPT-NeoX style
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    elif hasattr(model, "model") and hasattr(model.model, "gpt_neox") and hasattr(model.model.gpt_neox, "layers"):
        return list(model.model.gpt_neox.layers)
    
    # OPT/LLaMA-style decoders
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return list(model.decoder.layers)
    
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


# ============================================================================
# Memory Management
# ============================================================================


def _cleanup_gpu_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Chat Formatting Utilities
# ============================================================================


def format_prompts_from_strings(
    tokenizer: AutoTokenizer,
    prompt_strings: List[str],
) -> List[str]:
    """Format raw prompt strings using the tokenizer's chat template.
    
    Args:
        tokenizer: Tokenizer with chat template
        prompt_strings: List of user prompts
        
    Returns:
        List of formatted chat strings
    """
    formatted = []
    for prompt in prompt_strings:
        messages = [{"role": "user", "content": prompt}]
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return formatted


def format_prompts_from_pairs(
    tokenizer: AutoTokenizer,
    human_list: List[str],
    assistant_list: List[str],
) -> List[str]:
    """Format human-assistant conversation pairs using chat template.
    
    Args:
        tokenizer: Tokenizer with chat template
        human_list: List of user prompts
        assistant_list: List of assistant responses
        
    Returns:
        List of formatted conversation strings
    """
    formatted = []
    for human, assistant in zip(human_list, assistant_list):
        messages = [
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ]
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return formatted


def format_prompts_from_pairs_with_formatted_inputs(
    tokenizer: AutoTokenizer,
    formatted_human_list: List[str],
    assistant_list: List[str],
) -> List[str]:
    """Combine pre-formatted human inputs with assistant responses.
    
    Args:
        tokenizer: Tokenizer (for consistency, may not be used)
        formatted_human_list: Pre-formatted user prompts
        assistant_list: Assistant responses
        
    Returns:
        List of combined conversation strings
    """
    return [f"{human}{assistant}" for human, assistant in zip(formatted_human_list, assistant_list)]


# ============================================================================
# Core Activation Extraction
# ============================================================================


def _prepare_batch_inputs(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize and prepare batch inputs for the model.
    
    Args:
        tokenizer: Tokenizer to use
        prompts: List of text prompts
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of tokenized inputs (input_ids, attention_mask)
    """
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _generate_sequences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
) -> torch.Tensor:
    """Generate sequences from the model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        inputs: Tokenized input dictionary
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated token IDs
    """
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return outputs


def _decode_outputs(
    tokenizer: AutoTokenizer,
    outputs: torch.Tensor,
    verbose: bool = False,
) -> List[str]:
    """Decode generated token IDs to text.
    
    Args:
        tokenizer: Tokenizer to use
        outputs: Generated token IDs
        verbose: Whether to print decoded sequences
        
    Returns:
        List of decoded text strings
    """
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    if verbose:
        for i, text in enumerate(decoded):
            print(f"\n=== Decoded sequence {i} ===")
            print(text)
    return decoded


def _create_activation_hook(
    activations_dict: Dict[int, List[torch.Tensor]],
    layer_name: str,
    verbose: bool = False,
) -> callable:
    """Create a forward hook to capture activations.
    
    Args:
        activations_dict: Dictionary to store activations
        layer_name: Name/identifier of the layer
        verbose: Whether to print debug info
        
    Returns:
        Hook function
    """
    def hook(module, input, output):
        # output is typically a tuple; first element is the hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Detach and move to CPU to save memory
        activations_dict[layer_name].append(hidden_states.detach().cpu())
        
        if verbose:
            print(f"Captured activation from {layer_name}: {hidden_states.shape}")
    
    return hook


def _register_activation_hooks(
    model: AutoModelForCausalLM,
    layers_str: str,
    verbose: bool = False,
) -> Tuple[Dict[int, List[torch.Tensor]], List]:
    """Register forward hooks on specified layers.
    
    Args:
        model: The transformer model
        layers_str: String specifying which layers to hook
        verbose: Whether to print debug info
        
    Returns:
        Tuple of (activations_dict, hook_handles)
    """
    res_layers = get_res_layers_to_enumerate(model)
    total_layers = len(res_layers)
    layer_indices = parse_layers_arg(layers_str, total_layers)
    
    activations_dict = {idx: [] for idx in layer_indices}
    hook_handles = []
    
    for idx in layer_indices:
        hook = _create_activation_hook(activations_dict, idx, verbose)
        handle = res_layers[idx].register_forward_hook(hook)
        hook_handles.append(handle)
        
        if verbose:
            print(f"Registered hook on layer {idx}")
    
    return activations_dict, hook_handles


# ============================================================================
# High-Level Activation Extraction Functions
# ============================================================================


def get_batch_res_activations_with_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layers_str: str = "auto",
    max_length: int = 512,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[Dict[int, List[torch.Tensor]], List[str], int]:
    """Extract activations while generating new tokens.
    
    This function:
    1. Tokenizes prompts
    2. Generates responses
    3. Captures activations from specified layers during generation
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        prompts: List of input prompts
        layers_str: Which layers to extract ("auto", "all", or specific)
        max_length: Maximum input sequence length
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        verbose: Whether to print debug info
        
    Returns:
        Tuple of:
        - activations_dict: {layer_idx: [tensor of shape (seq_len, hidden_dim)] per example}
        - decoded_outputs: List of generated text
        - input_length: Length of input sequence
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Extracting activations with generation")
        print(f"Batch size: {len(prompts)}")
        print(f"Layers: {layers_str}")
        print(f"{'='*60}\n")
    
    # Prepare inputs
    inputs = _prepare_batch_inputs(tokenizer, prompts, max_length)
    input_length = inputs["input_ids"].shape[1]
    
    # Register hooks
    activations_dict, hook_handles = _register_activation_hooks(model, layers_str, verbose)
    
    try:
        # Generate with hooks active
        outputs = _generate_sequences(model, tokenizer, inputs, max_new_tokens, temperature)
        decoded_outputs = _decode_outputs(tokenizer, outputs, verbose)
        
    finally:
        # Always remove hooks
        for handle in hook_handles:
            handle.remove()
    
    # Process activations: convert list of batched tensors to list per example
    processed_activations = {}
    for layer_idx, activation_list in activations_dict.items():
        if len(activation_list) > 0:
            # Concatenate along sequence dimension if multiple forward passes
            # Typically there's one tensor per generation step
            # Shape: (batch_size, seq_len, hidden_dim)
            concatenated = torch.cat(activation_list, dim=1)
            
            # Split into list per example
            processed_activations[layer_idx] = [
                concatenated[i] for i in range(concatenated.shape[0])
            ]
        else:
            processed_activations[layer_idx] = []
    
    if verbose:
        print(f"\nExtracted activations from {len(processed_activations)} layers")
        for layer_idx, acts in processed_activations.items():
            if len(acts) > 0:
                print(f"  Layer {layer_idx}: {len(acts)} examples, shape {acts[0].shape}")
    
    return processed_activations, decoded_outputs, input_length


def get_batch_res_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    outputs: List[str],
    layers_str: str = "auto",
    max_length: int = 2048,
    verbose: bool = False,
) -> Tuple[Dict[int, List[torch.Tensor]], List[str], int]:
    """Extract activations from pre-generated sequences (no generation).
    
    This function is for when you already have complete sequences and just
    want to extract activations from a forward pass.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        outputs: List of complete sequences (already formatted)
        layers_str: Which layers to extract ("auto", "all", or specific)
        max_length: Maximum sequence length
        verbose: Whether to print debug info
        
    Returns:
        Tuple of:
        - activations_dict: {layer_idx: [tensor of shape (seq_len, hidden_dim)] per example}
        - outputs: The input sequences (passed through)
        - input_length: Length of input sequence
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Extracting activations from pre-generated sequences")
        print(f"Batch size: {len(outputs)}")
        print(f"Layers: {layers_str}")
        print(f"{'='*60}\n")
    
    # Tokenize the complete sequences
    inputs = tokenizer(
        outputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    
    input_length = inputs["input_ids"].shape[1]
    
    # Register hooks
    activations_dict, hook_handles = _register_activation_hooks(model, layers_str, verbose)
    
    try:
        # Single forward pass
        with torch.no_grad():
            _ = model(**inputs)
        
    finally:
        # Always remove hooks
        for handle in hook_handles:
            handle.remove()
    
    # Process activations: convert to list per example
    processed_activations = {}
    for layer_idx, activation_list in activations_dict.items():
        if len(activation_list) > 0:
            # Should be a single batched tensor from one forward pass
            # Shape: (batch_size, seq_len, hidden_dim)
            batched_activation = activation_list[0]
            
            # Split into list per example
            processed_activations[layer_idx] = [
                batched_activation[i] for i in range(batched_activation.shape[0])
            ]
        else:
            processed_activations[layer_idx] = []
    
    if verbose:
        print(f"\nExtracted activations from {len(processed_activations)} layers")
        for layer_idx, acts in processed_activations.items():
            if len(acts) > 0:
                print(f"  Layer {layer_idx}: {len(acts)} examples, shape {acts[0].shape}")
    
    return processed_activations, outputs, input_length

