This tool performs a deep architectural sweep of Transformer-based models (NLP and Vision) to extract every single weight matrix and bias vector. Unlike standard parameter counters, this script maps the **full tensor path** and **mathematical shapes** of every linear transformation in the model.

## How It Works

The extractor follows a three-stage process to map the model's internal geometry:

### 1. Model Loading & Detection

The script initializes models from **HuggingFace Transformers** or **Torchvision**. It identifies the model type (Causal LM, Encoder, or Vision Transformer) to correctly determine the maximum sequence length or patch count, which defines the "context shape" of the data flowing into each matrix.

### 2. Recursive Module Traversal

Instead of looking at a high-level summary, the tool uses `model.named_modules()` to visit every individual component. It specifically targets:

* `nn.Linear`: Standard dense layers found in BERT, ViT, and Mistral.
* `Conv1D`: The 1D convolution layers used in GPT-2 architectures as linear projections.
* `NonDynamicallyQuantizableLinear`: Specialized linear layers often found in attention mechanisms.

### 3. Metadata Extraction

For every layer identified, the tool extracts:

* **The Absolute Path**: The exact location in the hierarchy (e.g., `transformer.h.11.mlp.c_fc`).
* **Weight Geometry**: The  dimensions.
* **Bias Presence**: Whether the layer utilizes an additive bias vector and its shape.
* **Parameter Count**: Total elements () for that specific matrix.

---

## Output Format

The output is a structured JSON file organized by model name. Below is a breakdown of the schema:

### Root Object

| Key | Description |
| --- | --- |
| `total_models` | Number of models processed in the execution. |
| `models` | A dictionary where keys are model names and values are the model's data. |

### Model Object

Each entry in the `models` dictionary contains:

* `total_matrices`: The count of every linear transformation in the model.
* `max_sequence_length`: The token/patch limit (defines the middle dimension of the input).
* `matrices`: A list of objects, one for every matrix instance.

### Matrix Object Example

```json
{
  "layer_name": "transformer.h.0.mlp.c_fc",
  "input_shape": ["batch_size", 1024, 768],
  "weight_shape": [3072, 768],
  "has_bias": true,
  "bias_shape": [3072],
  "parameters": 2362368
}

```

#### Field Definitions:

* **`layer_name`**: The unique identifier for the matrix. The index (e.g., `.0.`) indicates which transformer block the matrix belongs to.
* **`input_shape`**: The shape of the activations entering the layer, usually represented as .
* **`weight_shape`**: The dimensions of the weight matrix itself.
* **`has_bias`**: A boolean flag for the presence of a bias vector.
* **`parameters`**: The sum of the weight elements and bias elements.

---

## Execution Summary

When the script completes, it prints a scannable table to the console:

```text
==========================================================================================
Model Name                     | Matrix Count    | Total Matrix Params
------------------------------------------------------------------------------------------
gpt2                           | 38              | 124,439,808
vit_b_16                       | 152             | 85,798,656
mistralai/Mistral-7B-v0.1      | 226             | 7,241,732,096
==========================================================================================

```

Would you like me to add a section to this README on how to use the generated JSON to calculate the theoretical memory bandwidth required for a forward pass?
