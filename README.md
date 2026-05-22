# Empirical Limits of Neuron-Level Ablation for AI Safety

This repository contains the code, data, and experimental analysis for our study on the limitations of neuron-level intervention in language models.

## Research Overview
Recent mechanistic interpretability research suggests that harmful behaviors (e.g., deception, strategic malice) may be localized to specific "malice-relevant" neurons. We empirically tested this hypothesis using GPT-2 as a transparent model system. 

Our findings indicate that:
1. **Harmful reasoning is distributed and polysemantic:** Malice-relevant units also participate in benign general knowledge (e.g., historical and coding tasks).
2. **Ablation causes systemic collapse:** Targeted removal of these units induces a "capability tax," leading to fragmented logic and stalling rather than clean safety.
3. **The Hydra Effect:** The model demonstrates architectural plasticity, dynamically re-routing harmful reasoning through redundant neural pathways post-ablation.

## Repository Structure
- `/data`: Contains the `robust_dataset_500.json` adversarial dataset.
- `/pipeline`:
    - `harvester.py`: Extracts latent activation manifolds from GPT-2.
    - `probe.py`: Trains sparse probes to identify malice-relevant circuitry.
    - `surgeon.py`: Implements surgical ablation of identified neurons.
    - `benchmark.py`: Evaluates the "Alignment Tax" on linguistic and logical capabilities.
- `/figures`: High-resolution plots documenting ablation failure, capability degradation, and polysemanticity analysis.

## Key Findings
- **Detection Accuracy:** Our sparse probes achieved 54.50% (± 2.45%) accuracy, suggesting that harmful intent is structurally entangled with standard reasoning.
- **Resilience:** Post-ablation, the model successfully re-routed rogue intent through alternative neural pathways, demonstrating that harmful intent is an emergent, adaptive property of the neural manifold rather than a static "tumor."

## Citation
If you use this research or code, please cite this work as:

```bibtex
@misc{Narayan_NeuronAblation,
  title={Empirical Limits of Neuron-Level Ablation for AI Safety},
  author={Narayan U},
  year={2026}
}
```

## Setup
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run the pipeline in order: `harvester.py` -> `probe.py` -> `surgeon.py` -> `benchmark.py`.

## Paper
- 📄 Paper: Available on ResearchGate (DOI: 10.13140/RG.2.2.19488.96001)
