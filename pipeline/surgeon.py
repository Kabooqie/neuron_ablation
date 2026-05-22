import torch
import numpy as np
import os
from transformer_lens import HookedTransformer

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDICES_PATH = "malice_indices.npy"
LAYERS_TO_SCAN = [7, 8, 9]


def load_and_ablate(indices_path):
    if not os.path.exists(indices_path):
        print(f"Error: {indices_path} not found. Run Probe.py first.")
        return None

    # 1. Load Model
    print(f"Loading {MODEL_NAME} and preparing for surgery...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    malice_indices = np.load(indices_path)

    # 2. Perform Surgery
    print(f"Ablating {len(malice_indices)} neurons identified as malice-relevant...")

    with torch.no_grad():
        for idx in malice_indices:
            # Map index back to layer/local_idx
            # Each layer has 768 neurons. We are using layers 7, 8, 9.
            layer_offset = idx // 768
            local_idx = idx % 768
            layer_num = LAYERS_TO_SCAN[layer_offset]

            # Surgery: Zero out output weights and bias
            model.blocks[layer_num].mlp.W_out[local_idx, :] = 0.0
            model.blocks[layer_num].mlp.b_out[local_idx] = 0.0

    print("Surgery complete. Malice pathways severed.")
    return model


if __name__ == "__main__":
    model = load_and_ablate(INDICES_PATH)
    if model:
        # Save the surgery for the benchmark script to load
        torch.save(model.state_dict(), "cured_model.pt")
        print("Cured model state saved as 'cured_model.pt'.")