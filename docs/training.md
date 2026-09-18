# Training

The end-to-end training pipeline projects the precomputed image embeddings into prefix tokens, concatenates them with text embeddings, and optimizes the decoder using next-token cross-entropy loss under teacher forcing.

## Training Command

To train the Photon model on MS-COCO, adjust the paths in `training/train_nano.py` to point to your precomputed embeddings and COCO annotations, then run:

```bash
python training/train_nano.py
```

### Verified Training Configuration

As verified in `paper.pdf` and `training/train_nano.py`:
- **Optimizer:** AdamW
- **Learning Rate:** 3e-4
- **Weight Decay:** 1e-2
- **Epochs:** 10
- **Batch Size:** 512
- **Precision:** AMP (FP16 compute, FP32 master weights)
- **Scheduler:** None
- **Gradient Clipping:** None

Checkpoints are saved to the `checkpoints/` directory after each epoch.
