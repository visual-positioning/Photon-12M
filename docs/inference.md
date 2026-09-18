# Inference

Photon uses greedy decoding for fast, autoregressive caption generation.

## Single Image / Demo

You can test inference using the benchmarking script or the live video demo:

```bash
python benchmarks/live_video_test.py
```
This requires a webcam or an RTSP stream (configurable in the script).

## Batch Inference

Batch inference for evaluation is performed in the benchmarking script:

```bash
python benchmarks/benchmark.py
```

This script will encode images using MobileCLIP, normalize the embedding, and autoregressively generate up to a maximum length of 32 tokens until the `[SEP]` token is emitted.
