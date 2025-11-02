# IGN-INF-Unlearning

Activation extraction and analysis toolkit for language model unlearning research.

### Python API

```python
from ign_inf_unlearning.models import get_model, get_batch_res_activations

# Load model
model, tokenizer = get_model("gpt2")

# Extract activations
activations, outputs, input_length = get_batch_res_activations(
    model=model,
    tokenizer=tokenizer,
    outputs=["Your text here", "Another sequence"],
    layers_str="auto",  # Automatically select layers
    verbose=True
)

# Access results
# activations: {layer_idx: [tensor per example]}
# Each tensor has shape (seq_len, hidden_dim)
```

### Command Line

**General text extraction:**
```bash
# Basic extraction (every 3rd layer by default)
python scripts/extraction/extract_activations.py \
    --model gpt2 \
    --data data/raw/my_data.jsonl \
    --output-dir data/activations/exp1

# Specify specific layers
python scripts/extraction/extract_activations.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --data data.jsonl \
    --output-dir data/activations/exp2 \
    --layers 0,5,10,15,20

# With text generation
python scripts/extraction/extract_activations.py \
    --model gpt2 \
    --data prompts.jsonl \
    --output-dir data/activations/exp3 \
    --with-generation \
    --max-new-tokens 50
```

**WMDP-BIO-FORGET MCQA extraction:**
```bash
# Extract from all WMDP datasets
python scripts/extraction/extract_wmdp_activations.py \
    --model gpt2 \
    --data-dir data/datasets/input_data/wmdp-bio-forget-mcqa \
    --output-dir data/activations/wmdp_exp1

# Extract from specific datasets only
python scripts/extraction/extract_wmdp_activations.py \
    --model gpt2 \
    --data-dir data/datasets/input_data/wmdp-bio-forget-mcqa \
    --output-dir data/activations/wmdp_exp2 \
    --datasets robust gibberish
```

**Output structure:**
```
data/activations/exp1/
  ├── dataset.csv      # Text sequences (and responses if generating)
  ├── layer_0.pkl      # Activations for layer 0 (list of tensors)
  ├── layer_3.pkl      # Activations for layer 3
  └── layer_6.pkl      # etc.

# For WMDP datasets (multiple subdirectories):
data/activations/wmdp_exp1/
  ├── robust/          # wmdp_bio_robust.csv
  │   ├── dataset.csv
  │   ├── layer_0.pkl
  │   └── layer_3.pkl
  ├── gibberish/       # gibberish.csv
  │   └── ...
  └── ...
```

## Project Structure (incomplete)

```
ign-inf-unlearning/
├── src/ign_inf_unlearning/
│   ├── models/
│   │   ├── activations.py      # Core activation extraction utilities
│   │   ├── inference.py         # Model inference utilities
│   │   └── probes.py           # Probe training utilities
│   ├── data/
│   │   ├── loaders.py          # Data loading utilities
│   │   └── rewriting.py        # Data rewriting utilities
│   ├── analysis/
│   │   ├── evaluation.py       # Evaluation metrics
│   │   ├── signatures.py       # Activation signature analysis
│   │   └── visualization.py    # Plotting utilities
│   └── utils/
│       ├── io.py               # I/O utilities
│       └── logging.py          # Logging utilities
├── scripts/
│   ├── extraction/
│   │   └── extract_activations.py  # CLI for activation extraction
│   ├── data/
│   │   ├── prepare_dataset.py      # Dataset preparation
│   │   └── rewrite_questions.py    # Question rewriting
│   ├── analysis/
│   │   ├── analyze_activations.py  # Activation analysis
│   │   └── evaluate_wmdp-bio-forget.py
│   └── probes/
│       ├── train_probes.py         # Train classification probes
│       └── compare_probes.py       # Compare probe performance
├── examples/
│   └── extract_activations_example.py  # Usage examples
├── tests/                      # Unit tests
├── data/                       # Data directory
├── experiments/               # Experiment configs and notebooks
└── docs/                      # Documentation
```

## Core Functions

### Model & Layer Utilities

- `get_model(model_name)`: Load model and tokenizer from HuggingFace
- `parse_layers_arg(layers_str, total_layers)`: Parse layer specifications
- `get_res_layers_to_enumerate(model)`: Get residual stream layers from model

### Activation Extraction

- `get_batch_res_activations(...)`: Extract activations from sequences
- `get_batch_res_activations_with_generation(...)`: Extract while generating text
- `format_prompts_from_strings(...)`: Format prompts using chat templates

## Layer Selection

The activation extraction supports flexible layer specification:

- `"auto"` (default): Every third layer (0, 3, 6, 9, ...)
- `"all"`: Extract from all layers
- `"0,5,10"`: Specific comma-separated indices
- `"0-10"`: Range (inclusive)
- `"0-10:2"`: Range with step (every 2nd layer)

### Reading Extracted Activations

```python
import pickle
import pandas as pd

# Load dataset
df = pd.read_csv("data/activations/exp1/dataset.csv")

# Load activations for a specific layer
with open("data/activations/exp1/layer_0.pkl", "rb") as f:
    layer_0_activations = pickle.load(f)
    # layer_0_activations is a list of tensors, one per example
    # Each tensor has shape (seq_len, hidden_dim)

print(f"Dataset has {len(df)} examples")
print(f"Layer 0 has {len(layer_0_activations)} activation tensors")
print(f"First example activation shape: {layer_0_activations[0].shape}")
```

This toolkit was built on top of the HuggingFace Transformers library.

