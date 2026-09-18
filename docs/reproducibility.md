# Reproducibility Notes & Discrepancies

This repository has been updated to reflect the official `paper.pdf` results. 

## Paper vs Old Repository Notes

Earlier draft versions of the repository README contained differing, preliminary values that have since been superseded by the formal publication:

1. **GFLOPs:** The old README claimed 0.49 GFLOPs. The official paper correctly reports 3.72 Total GFLOPs (3.57 Vision GFLOPs + 0.15 Decoder GFLOPs).
2. **FPS / Throughput:** The old README claimed 942.3 FPS. The official paper reports 9.09 FPS (total inference including vision latency on GPU) and ~106 decoder tokens/second at batch size 1. The 942+ FPS metric likely referred to isolated token decoding throughput at very large batch sizes (the paper notes decoding throughput scales up to 82,000+ tokens/sec at batch size 512).
3. **CIDEr Scale:** The old README reported CIDEr as 1.088. The paper reports it as 108.59, which is the standard 100x scaling used by the COCO evaluation toolkit.

## Verification

All numbers presented in the new README and `docs/` reflect the exact values published in `paper.pdf`.
