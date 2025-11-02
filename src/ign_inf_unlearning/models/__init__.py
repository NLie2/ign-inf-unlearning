"""Models module for loading and working with language models."""

from .activations import (
    _cleanup_gpu_memory,
    format_prompts_from_pairs,
    format_prompts_from_pairs_with_formatted_inputs,
    format_prompts_from_strings,
    get_batch_res_activations,
    get_batch_res_activations_with_generation,
    get_model,
    get_pad_token,
    get_res_layers_to_enumerate,
    parse_layers_arg,
    suggest_layer,
)

__all__ = [
    # Model utilities
    "get_model",
    "get_pad_token",
    "get_res_layers_to_enumerate",
    "parse_layers_arg",
    "suggest_layer",
    # Activation extraction
    "get_batch_res_activations",
    "get_batch_res_activations_with_generation",
    # Formatting utilities
    "format_prompts_from_strings",
    "format_prompts_from_pairs",
    "format_prompts_from_pairs_with_formatted_inputs",
    # Memory management
    "_cleanup_gpu_memory",
]

