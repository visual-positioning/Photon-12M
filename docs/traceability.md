# Paper ↔ Code Traceability Table

| Paper Component | Paper Location | Code Location | Config | Command |
|---|---|---|---|---|
| MobileCLIP-S1 encoder | Section 3.1 | `architecture/mobilecap_modern.py` | `mobileclip_s1.pt` | N/A (Frozen) |
| Prefix projection | Section 3.1 | `architecture/mobilecap_modern.py` | 512->2048->2048 | N/A |
| Transformer decoder | Section 3.1 | `architecture/mobilecap_modern.py` | 6 layers, d=256, h=8 | N/A |
| Tokenizer | Section 4.1 | `tokenizer/coco2014_tokenizer.json` | 8000 vocab size | N/A |
| Training | Section 3.2 | `training/train_nano.py` | BS=512, LR=3e-4 | `python training/train_nano.py` |
| Evaluation | Section 4.3 | `benchmarks/benchmark.py` | Greedy, max_tokens=20 | `python benchmarks/benchmark.py` |
| Benchmarking | Section 5.2 | `benchmarks/benchmark.py` | `estimate_flops()` | `python benchmarks/benchmark.py` |
