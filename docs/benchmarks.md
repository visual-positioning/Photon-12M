# Benchmarking

Efficiency experiments in `paper.pdf` measure GPU and CPU latency, decoding speed, and GFLOPs.

## Running the Benchmark

The `benchmarks/benchmark.py` script performs inference over the MS-COCO validation set and computes latency, throughput, and FLOPs.

```bash
python benchmarks/benchmark.py
```

### Metrics Reported

* **Latency (ms/img):** Total inference time per image.
* **Decode (token/s):** Throughput of the autoregressive decoder.
* **FPS:** Frames processed per second.
* **GFLOPs:** Calculated as `2 * total_params * context_len` in the script.

*Note: The paper reports 3.72 total GFLOPs (3.57 Vision + 0.15 Decoder), while the script may estimate FLOPs slightly differently. Always refer to `paper.pdf` for official published values.*
