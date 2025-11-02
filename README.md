# IGN-INF-Unlearning

Activation extraction and analysis toolkit for language model unlearning research.

## Features

- **Activation Extraction**: Extract residual stream activations from transformer models
- **Simple Output Format**: CSV for dataset + pickle files per layer
- **Flexible Data Loading**: Support for JSONL and CSV input formats
- **Batch Processing**: Efficient GPU memory management for large datasets
- **Default Layer Selection**: Every third layer by default
- **CLI and Python API**: Use from command line or integrate into your code
- **Model Support**: Works with Llama, GPT, and other transformer architectures

## Installation

```bash
# Clone the repository
cd /rds/general/user/nk1924/home/ign-inf-unlearning

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Jupyter Notebook Setup

To use the project environment in Jupyter notebooks:

```bash
# Install ipykernel if not already installed
uv pip install ipykernel

# Register the environment as a Jupyter kernel
uv run python -m ipykernel install --user --name=ign-inf-unlearning --display-name="Python (ign-inf-unlearning)"
```

After running these commands, the kernel will be available in Jupyter/VS Code notebook kernel selector as **"Python (ign-inf-unlearning)"**.

## Quick Start

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

## Examples

- `examples/simple_extraction_example.py` - Basic activation extraction
- `examples/wmdp_extraction_example.py` - WMDP-BIO-FORGET MCQA extraction

## Project Structure

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

### Memory Management

- `_cleanup_gpu_memory()`: Clear CUDA cache and run garbage collection

## Layer Selection

The activation extraction supports flexible layer specification:

- `"auto"` (default): Every third layer (0, 3, 6, 9, ...)
- `"all"`: Extract from all layers
- `"0,5,10"`: Specific comma-separated indices
- `"0-10"`: Range (inclusive)
- `"0-10:2"`: Range with step (every 2nd layer)

## Data Formats

### Input Formats

**JSONL**:
```json
{"text": "Your sequence here"}
{"text": "Another sequence"}
```

**CSV**:
```csv
text,label
"First sequence",positive
"Second sequence",negative
```

### Output Format

Always saves as:
- **CSV file** (`dataset.csv`): Contains text sequences and optional generated responses
- **Pickle files per layer** (`layer_N.pkl`): Each contains a list of activation tensors

## Advanced Usage

### Processing Large Datasets

```python
from tqdm import tqdm
from ign_inf_unlearning.models import (
    get_model, 
    get_batch_res_activations, 
    _cleanup_gpu_memory
)

model, tokenizer = get_model("gpt2")
batch_size = 32

for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i+batch_size]
    
    activations, _, _ = get_batch_res_activations(
        model=model,
        tokenizer=tokenizer,
        outputs=batch,
        layers_str="auto"
    )
    
    # Process results...
    
    # Critical: cleanup after each batch
    del activations
    _cleanup_gpu_memory()
```

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

## Supported Models

The toolkit supports various transformer architectures:

- **Llama family** (Llama-2, Llama-3, etc.)
- **GPT family** (GPT-2, GPT-Neo, GPT-J)
- **Other transformers** with standard architectures

## Performance Tips

1. **Batch size**: Start with 32, adjust based on GPU memory
2. **Layer selection**: Default `"auto"` extracts every 3rd layer (good balance)
3. **Max length**: Reduce to 1024 or 512 if running out of memory
4. **FP16**: Models automatically use half precision on GPU

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 8

# Reduce max length
--max-length 1024

# Use fewer layers
--layers 0,6,12
```

### Model Not Found

```python
# For gated models, login first
from huggingface_hub import login
login()
```

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src/ign_inf_unlearning

# Run specific test file
pytest tests/test_models/test_activations.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ign_inf_unlearning,
  title = {IGN-INF-Unlearning: Activation Extraction for Language Model Unlearning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ign-inf-unlearning}
}
```

## License

[Your License Here]

## Contact

[Your Contact Information]

## Acknowledgments

This toolkit was built on top of the HuggingFace Transformers library.

