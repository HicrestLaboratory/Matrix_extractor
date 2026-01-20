# Transformer Matrix Extractor

A Python tool for extracting and analyzing all weight matrices from transformer models (GPT, BERT, ViT, Mistral, etc.). This tool provides detailed information about every linear layer in your model, including weight shapes, bias information, and parameter counts.

## Features

- Extracts **all** weight matrices from transformer models
- Supports multiple model architectures (GPT-2, BERT, ViT, Mistral)
- Preserves full layer paths for precise identification
- Exports results to JSON for further analysis
- Provides summary statistics

## Installation

```bash
pip install torch transformers torchvision
```

## Usage

### Basic Usage (Default Models)

Run the extractor with default models:

```bash
python transformer_extractor.py
```

This will analyze the following models:
- GPT-2 (all variants: base, medium, large, xl)
- Vision Transformer (vit_b_16, vit_l_16, vit_h_14)
- Mistral-7B
- BERT (base and large)

### Custom Models

Specify which models to analyze:

```bash
python transformer_extractor.py --models gpt2 bert-base-uncased
```

### Custom Output File

Change the output JSON filename:

```bash
python transformer_extractor.py --output my_analysis.json
```

### Combined Options

```bash
python transformer_extractor.py --models gpt2 gpt2-medium --output gpt2_matrices.json
```

## Output Format

### Console Output

The tool prints a summary table:

```
==========================================================================================
Model Name                     | Matrix Count    | Total Matrix Params
------------------------------------------------------------------------------------------
gpt2                           | 148             | 124,439,808
bert-base-uncased              | 199             | 109,482,240
==========================================================================================
```

### JSON Output

The JSON file contains detailed information for each matrix:

```json
{
  "total_models": 2,
  "models": {
    "gpt2": {
      "total_matrices": 148,
      "matrices": [
        {
          "layer_name": "transformer.wte",
          "weight_shape": [50257, 768],
          "has_bias": false,
          "bias_shape": null,
          "parameters": 38597376
        },
        {
          "layer_name": "transformer.h.0.attn.c_attn",
          "weight_shape": [768, 2304],
          "has_bias": true,
          "bias_shape": [2304],
          "parameters": 1771776
        }
      ]
    }
  }
}
```

### Output Fields Explained

- **layer_name**: Full path to the layer in the model (e.g., `transformer.h.0.attn.c_attn`)
- **weight_shape**: Dimensions of the weight matrix `[out_features, in_features]`
- **has_bias**: Boolean indicating if the layer has a bias term
- **bias_shape**: Shape of the bias vector (if present)
- **parameters**: Total parameter count for this layer (weights + bias)

## Understanding the Results

### Weight Shape Format

Weight matrices are stored as `[out_features, in_features]`:
- **out_features**: Number of output dimensions
- **in_features**: Number of input dimensions

For example, a shape of `[2304, 768]` means:
- Takes 768-dimensional input
- Produces 2304-dimensional output

### Common Layer Types

**Attention Layers** (e.g., `h.0.attn.c_attn`):
- Combined Q, K, V projection in GPT-2
- Shape: `[hidden_dim, 3 * hidden_dim]`

**Feed-Forward Layers** (e.g., `h.0.mlp.c_fc`):
- MLP expansion layer
- Typically `[hidden_dim, 4 * hidden_dim]`

**Output Projections** (e.g., `h.0.attn.c_proj`):
- Projects back to model dimension
- Shape: `[hidden_dim, hidden_dim]`

## Use Cases

1. **Model Analysis**: Understand the architecture and parameter distribution
2. **Research**: Study how different models structure their weights
3. **Optimization**: Identify large matrices for compression or pruning
4. **Education**: Learn transformer architecture by examining real models
5. **Debugging**: Verify model structure matches expectations

## Supported Models

### Language Models
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- BERT family (bert-base-uncased, bert-large-uncased)
- Mistral (mistralai/Mistral-7B-v0.1)
- Any HuggingFace transformer model

### Vision Models
- Vision Transformer (vit_b_16, vit_l_16, vit_h_14)

### Adding Custom Models

Modify the `DEFAULT_MODELS` list in the code or pass custom model names via `--models`:

```python
DEFAULT_MODELS = [
    "your-model-name",
    "organization/model-name",
]
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (HuggingFace)
- TorchVision (for ViT models)

## Performance Notes

- Large models (like Mistral-7B) may take several minutes to load
- Models are loaded on CPU by default
- JSON files can be large for models with many layers

## Troubleshooting

**Model fails to load**: Some models may require authentication or special access. Use HuggingFace CLI to login:
```bash
huggingface-cli login
```

**Out of memory**: For very large models, ensure you have sufficient RAM (16GB+ recommended for 7B+ models)

**Missing dependencies**: Install all requirements:
```bash
pip install torch transformers torchvision
```