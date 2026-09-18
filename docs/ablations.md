# Ablation Studies

Photon includes extensive ablation studies validating its architectural choices.

## Running Ablations

You can reproduce the ablations using the training ablation scripts:

```bash
python training/train_ablations.py
```
*(Configured via `config/ablation_config.yaml`)*

## Key Findings (from `paper.pdf`)

* **Prefix Length (K):** $K=8$ achieves the best balance. $K=16$ yields negligible improvements at higher computational cost.
* **Decoder Depth:** 6 layers is optimal. Moving to 8 layers provides only marginal gains.
* **Conditioning Strategy:** Prefix-only conditioning consistently outperforms single-token, mean-pooled, and lightweight cross-attention approaches.
* **Precision:** Mixed-precision FP16 decoding matches FP32 performance while accelerating inference.
