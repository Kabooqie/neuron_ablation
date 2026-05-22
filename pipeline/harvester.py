import torch
import numpy as np
import json
from transformer_lens import HookedTransformer
import os

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
DATASET_PATH = "robust_dataset_500.json"
LAYERS_TO_SCAN = [7, 8, 9]
OUTPUT_X = "X_robust.npy"
OUTPUT_Y = "y_robust.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading {MODEL_NAME} to {DEVICE}...")
    return HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)


def harvest_activations(model, dataset):
    X_list, y_list = [], []
    print(f"Harvesting activations from {len(dataset)} samples...")

    for item in dataset:
        for prompt, label in [(item["aligned_prompt"], 0), (item["rogue_prompt"], 1)]:
            # Truncate to ensure the model doesn't crash on long sequences
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, -1024:]

            _, cache = model.run_with_cache(tokens)

            # Concatenate middle layers (L7, L8, L9) into one vector
            combined_act = np.concatenate([
                cache[f"blocks.{l}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
                for l in LAYERS_TO_SCAN
            ])
            X_list.append(combined_act)
            y_list.append(label)

    return np.array(X_list), np.array(y_list)


if __name__ == "__main__":
    # Ensure data exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
    else:
        model = load_model()
        with open(DATASET_PATH, "r") as f:
            sandbox_data = json.load(f)

        X, y = harvest_activations(model, sandbox_data)

        np.save(OUTPUT_X, X)
        np.save(OUTPUT_Y, y)

        print(f"\nSuccess!")
        print(f"Harvested {X.shape[0]} vectors.")
        print(f"Dimensions: {X.shape[1]} (Concatenated L7, L8, L9).")