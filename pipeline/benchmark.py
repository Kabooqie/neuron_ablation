import torch
import numpy as np
import os
from transformer_lens import HookedTransformer

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURED_MODEL_PATH = "cured_model.pt"
MALICE_INDICES_PATH = "malice_indices.npy"

# Diverse Benchmark Suite
BENCHMARKS = {
    "Coherence": ["The capital of France is", "To be or not to be, that is the", "The history of the Roman Empire began in"],
    "Factual": ["The primary language spoken in Brazil is", "The sun rises in the", "Water boils at"],
    "Logic": ["If all cats have tails and my pet is a cat, then", "144 divided by 12 multiplied by 5 is", "Write a python loop:"]
}

def get_stalling_rate(model, prompts):
    """Measures 'Stalling' (incoherence/fragmentation) as a percentage."""
    stalled = 0
    for p in prompts:
        out = model.generate(p, max_new_tokens=15, verbose=False)
        # Logic: If output is < 5 words, it stalled or fragmented
        if len(out.split()) < 5:
            stalled += 1
    return (stalled / len(prompts)) * 100

def run_evaluation(model, label):
    print(f"\n--- Running Evaluation: {label} ---")
    results = {}
    for name, prompts in BENCHMARKS.items():
        rate = get_stalling_rate(model, prompts)
        results[name] = rate
        print(f"{name} Stalling Rate: {rate:.1f}%")
    return results

# 1. Load Baseline
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
baseline_results = run_evaluation(model, "BASELINE (Pre-Surgery)")

# 2. Perform Surgery
malice_indices = np.load(MALICE_INDICES_PATH)
with torch.no_grad():
    for idx in malice_indices:
        layer_num = 7 + (idx // 768)
        local_idx = idx % 768
        model.blocks[layer_num].mlp.W_out[local_idx, :] = 0.0
        model.blocks[layer_num].mlp.b_out[:] = 0.0

# 3. Evaluate Cured Model
cured_results = run_evaluation(model, "CURED (Post-Surgery)")

# 4. Final Comparison
print("\n--- FINAL BENCHMARK SUMMARY ---")
for name in BENCHMARKS.keys():
    diff = cured_results[name] - baseline_results[name]
    print(f"{name}: Stalling increased by {diff:.1f}%")