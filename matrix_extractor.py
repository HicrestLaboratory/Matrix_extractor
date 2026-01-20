import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
from torchvision import models
from typing import List, Dict
import warnings
import json
import argparse

warnings.filterwarnings("ignore")

class FullTransformerMatrixExtractor:
    """
    Extracts EVERY matrix from transformer models.
    Preserves the full path for every layer across all blocks.
    """

    DEFAULT_MODELS = [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "vit_b_16", "vit_l_16", "vit_h_14",
        "mistralai/Mistral-7B-v0.1",
        "bert-base-uncased", "bert-large-uncased",
    ]

    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or self.DEFAULT_MODELS
        self.models = {}
        self._load_models()

    def _load_models(self):
        print(f"Loading {len(self.model_names)} models...")
        for name in self.model_names:
            print(f"  Loading {name}...")
            try:
                if name.startswith('vit_'):
                    model = getattr(models, name)(weights=None)
                else:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(name)
                    except:
                        model = AutoModel.from_pretrained(name)
                
                self.models[name] = model
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")

    def _is_linear_layer(self, module) -> bool:
        return isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)) or \
               module.__class__.__name__ == 'Conv1D'

    def extract_all_matrices(self) -> Dict:
        """Extracts every instance of a weight matrix in the model."""
        models_data = {}
        
        for model_name, model in self.models.items():
            matrices = []
            
            # Iterate through all modules without deduplication
            for layer_name, module in model.named_modules():
                if not self._is_linear_layer(module):
                    continue
                
                weight = module.weight
                out_features, in_features = weight.shape
                
                has_bias = module.bias is not None
                bias_shape = list(module.bias.shape) if has_bias else None
                
                matrices.append({
                    "layer_name": layer_name,
                    "weight_shape": [out_features, in_features],
                    "has_bias": has_bias,
                    "bias_shape": bias_shape,
                    "parameters": weight.numel() + (module.bias.numel() if has_bias else 0),
                })
            
            models_data[model_name] = {
                "total_matrices": len(matrices),
                "matrices": matrices
            }
        
        return models_data

    def save_to_json(self, path: str):
        data = {
            "total_models": len(self.models),
            "models": self.extract_all_matrices(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Saved ALL matrices to {path}")

    def print_summary(self):
        models_data = self.extract_all_matrices()
        print("\n" + "="*90)
        print(f"{'Model Name':<30} | {'Matrix Count':<15} | {'Total Matrix Params'}")
        print("-" * 90)
        for name, data in models_data.items():
            total_params = sum(m["parameters"] for m in data["matrices"])
            print(f"{name:<30} | {data['total_matrices']:<15} | {total_params:,}")
        print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", default="all_transformer_matrices.json")
    args = parser.parse_args()
    
    extractor = FullTransformerMatrixExtractor(model_names=args.models)
    extractor.print_summary()
    extractor.save_to_json(args.output)