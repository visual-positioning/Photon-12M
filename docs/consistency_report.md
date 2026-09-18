# Documentation Consistency Report

## Verified from `paper.pdf`
- **Scientific facts:** Prefix-conditioned multimodal architecture without cross-attention.
- **Published results:** 12.41M trainable params, 33.91M total params, 3.72 GFLOPs total, 108.59 CIDEr on COCO.
- **Experimental setup:** Batch size 512, 10 epochs, 3e-4 LR, 1e-2 WD, AMP, MobileCLIP-S1 encoder, K=8 prefixes, 6-layer decoder.

## Verified from Code
- **Commands:** Actual training and evaluation run via `train_nano.py`, `evaluate_tokenizers.py`, and `benchmark.py`.
- **Configs:** Model construction in `mobilecap_modern.py` perfectly aligns with the 6-layer, 256-dim, 8-head Nano-LLaMA architecture described.
- **Implementation:** Precomputed embeddings logic, BPE tokenizer handling, and greedy decoding logic are fully present and functional.

## Discrepancies
- The original repository README featured draft metrics (0.49 GFLOPs, 942.3 FPS, 1.088 CIDEr) which conflicted with the formalized published results in `paper.pdf`. These have been corrected in the updated documentation.
- SPICE evaluation was found to be explicitly disabled in the code (`benchmark.py`) to prevent Java crashes, though it is reported in the paper.

## Not Reproducible Directly
- **Cross-dataset evaluation (Flickr30k, NoCaps, TextCaps):** While scripts like `extract_flickr30k_karpathy_embeddings.py` exist, the end-to-end evaluation commands for out-of-domain datasets are not fully unified in a single simple command, relying instead on manual path adjustments in `evaluate_tokenizers.py`.

## Missing Assets
- **Checkpoints:** Only `nano_ep55.pt` exists in the local directory; the official `nano_ep10.pt` representing the exact paper state is not pre-packaged.
- **Datasets:** Users must download MS-COCO manually.

## Remaining Documentation Gaps
- Fully automated reproduction scripts for ablations exist, but require manual YAML editing to test all variants from the paper.
