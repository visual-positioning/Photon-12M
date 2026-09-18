# FAQ

**Q: Does Photon use explicit cross-attention?**
A: No. Modalities interact through a unified self-attention mechanism applied to the concatenated sequence of visual prefix tokens and text tokens.

**Q: Can I use a different vision encoder?**
A: Yes, but you will need to adjust the projection MLP dimensions in `architecture/mobilecap_modern.py` to match the new encoder's output dimension, and retrain the model.

**Q: Why use a vocabulary size of 8,000?**
A: A compact vocabulary size, trained specifically on the COCO Karpathy split, prevents linguistic leakage and reduces the parameter count of the tied language modeling head, maintaining the model's nano-scale footprint.

**Q: Is there support for multi-GPU training?**
A: The current `train_nano.py` script is set up for single-GPU training. However, the lightweight nature of the model means it easily trains on a single consumer GPU (e.g., RTX 4060) in just ~16 minutes.
