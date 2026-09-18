# Troubleshooting

### Missing MobileCLIP Weights
**Error:** File not found for `models/mobileclip_s1.pt`.
**Solution:** Download the weights manually via `wget -O models/mobileclip_s1.pt https://huggingface.co/apple/MobileCLIP-S1/resolve/main/mobileclip_s1.pt` or use Docker.

### METEOR / SPICE Evaluation Errors (Java)
**Error:** Crashes during COCO Eval when calculating METEOR or SPICE.
**Solution:** Ensure Java 11 (`openjdk-11-jre`) is installed. In `benchmarks/benchmark.py`, SPICE has been commented out to prevent known crashes with Java 21. Use Java 11 for full compatibility.

### Missing Precomputed Embeddings
**Error:** `CachedCocoDataset` warning that embeddings are missing.
**Solution:** Run the `datapreprocessing/extract_coco2014_karapathy_embedding.py` script first to generate `.pt` embeddings from your dataset.

### OpenCV / FFmpeg Issues
If running live video inference fails, ensure you have `ffmpeg` and `libgl1` installed on your system (these are included in the Dockerfile).
