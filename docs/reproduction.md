# Experiment Reproduction Matrix

| Experiment | Paper Section/Table | Code | Command | Expected Output |
|---|---|---|---|---|
| MS-COCO main results | Table 2 | `benchmarks/benchmark.py` | `python benchmarks/benchmark.py` | CIDEr ~108.59, B@4 ~32.33 |
| GPU latency benchmark | Table 5 | `benchmarks/benchmark.py` | `python benchmarks/benchmark.py` | ~19.95ms vision, ~109.96ms total |
| Ablation Study (Depth/Width) | Table 20/21 | `training/train_ablations.py` | `python training/train_ablations.py` | Varies by YAML config |
| Cross-Dataset Evaluation | Tables 7-12 | `evaluation/evaluate_tokenizers.py` | *Paper result; reproduction command not fully exposed via CLI without manual path edits* | Varies by dataset |
| Batch Scaling | Table 17 | `benchmarks/speedtest.py` | `python benchmarks/speedtest.py` | Near-linear throughput scaling |
