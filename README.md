# Photon: Efficient Prefix-Conditioned Image Captioning with Lightweight Transformer Decoding

This is the official repository for:

**Photon: Efficient Prefix-Conditioned Image Captioning with Lightweight Transformer Decoding**

**Authors:** Lakshmi Ganapathi Kodi, Dharmendra Chauhan, Dhanush Reddy Gangireddy, Sonam Kumari Chaudhary, Kalidas Yeturu, Nikhil Srivastava

**Published in:** Machine Learning (Springer)

📄 **Paper:** [Link](https://link.springer.com/article/10.1007/s10994-026-07072-4) | 💻 **Code:** [GitHub](https://github.com/visual-positioning/Photon-12M)

---

### News
* **[2026.09.18]** Official Photon implementation and evaluation code released.
* **[2026.07.15]** Photon published in *Machine Learning* (Springer).

---

### Highlights

* **Lightweight prefix-conditioned multimodal captioning**
* **Frozen MobileCLIP-S1 visual encoder**
* **8 visual prefix tokens**
* **Compact 6-layer decoder-only Transformer**
* **RoPE + RMSNorm + SwiGLU**
* **12.41M trainable parameters**
* **33.91M total parameters**
* **3.72 GFLOPs**
* **CIDEr 108.59 on MS-COCO Karpathy split**
* **2.41× GPU speed-up over BLIP**
* **8.28× CPU speed-up over BLIP**

---

## Overview

Image captioning systems based on large vision–language models often rely on computationally intensive cross-attention architectures, limiting their suitability for real-time and resource-constrained deployment. To address this, we propose **Photon**, a lightweight prefix-conditioned multimodal captioning framework. 

Photon combines a frozen MobileCLIP-S1 vision encoder with a compact decoder-only Transformer. Visual information is injected through a small set of learned prefix tokens, enabling efficient multimodal conditioning without region-based processing or heavy cross-modal attention. The architecture extracts a 512-dimensional global image embedding, applies L2 normalization, and projects it into 8 visual prefix tokens. These are concatenated with text embeddings and processed by a 6-layer decoder-only Transformer (incorporating RoPE, RMSNorm, and SwiGLU) for autoregressive caption generation.

By avoiding explicit cross-attention entirely and replacing fine-grained multimodal fusion with efficient global prefix conditioning, Photon significantly reduces computational overhead while maintaining competitive visual grounding and caption quality.

### Architecture Configuration

| Component | Configuration |
|---|---|
| Vision Encoder | MobileCLIP-S1 |
| Image Embedding | 512 |
| Prefix Tokens | 8 |
| Decoder Layers | 6 |
| Hidden Dimension | 256 |
| Attention Heads | 8 |
| Head Dimension | 32 |
| FFN | SwiGLU |
| FFN Intermediate Dimension | 768 |
| Positional Encoding | RoPE |
| Normalization | RMSNorm |
| Vocabulary Size | 8,000 |
| Maximum Caption Length | 32 |

---

## Getting Started

### Environment Setup

The recommended and most robust way to run Photon is using the provided Docker environment, which handles all dependencies including Java (required for COCO metrics) and CUDA.

**Using Docker:**
```bash
# Build the image (automatically downloads MobileCLIP-S1 weights)
docker build -t mobilecap .

# Run the container with GPU support
docker run --gpus all -it -v $(pwd):/workspace mobilecap
```

**Native Installation:**
If you prefer a native installation, ensure you have Python 3, PyTorch 2.0+, and Java 11 installed.

```bash
pip install -r requirements.txt

# Manually download the MobileCLIP-S1 weights
mkdir -p models
wget -O models/mobileclip_s1.pt https://huggingface.co/apple/MobileCLIP-S1/resolve/main/mobileclip_s1.pt
```

---

## Data Preparation

Photon uses the MS-COCO 2014 dataset with the Karpathy split for training and evaluation.

1. **Download MS-COCO 2014** (Train and Val images).
2. **Download the Karpathy split JSON** annotations.
3. Update the paths in `datapreprocessing/extract_coco2014_karapathy_embedding.py` to point to your local MS-COCO directory.

To accelerate training, Photon relies on precomputed L2-normalized MobileCLIP-S1 image embeddings. Extract the embeddings by running:

```bash
python datapreprocessing/extract_coco2014_karapathy_embedding.py
```
This will generate `.pt` files in your dataset's `embeddings/` directory.

---

## Training

To train the Photon model on the MS-COCO dataset, adjust the dataset and embedding paths inside `training/train_nano.py` and run:

```bash
python training/train_nano.py
```
**Training details:** The model trains with a batch size of 512 using the AdamW optimizer (LR=3e-4, Weight Decay=1e-2) for 10 epochs using Automatic Mixed Precision (AMP). Checkpoints are saved automatically to the `checkpoints/` directory.

---

## Evaluation & Benchmarking

### Benchmarking (Throughput and COCO Metrics)
To evaluate the model's inference latency, throughput, FLOPs, and standard captioning metrics (CIDEr, BLEU-4, METEOR, ROUGE-L) on the MS-COCO validation set, ensure the checkpoint paths are set correctly in `benchmarks/benchmark.py` and run:

```bash
python benchmarks/benchmark.py
```

### Cross-Dataset Tokenizer Evaluation
To evaluate the model's semantic alignment metrics and zero-shot performance across datasets (requires updating local paths in the script), run:

```bash
python evaluation/evaluate_tokenizers.py
```

---

## Results

### MS-COCO Main Results

*(Scores obtained on the MS-COCO Karpathy test split)*

| Model | B@1 | B@2 | B@3 | B@4 | CIDEr | SPICE | METEOR | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| ClipCap | 70.96 | 54.36 | 41.25 | 31.49 | 106.72 | 20.51 | 27.72 | 54.85 |
| SmallCap | 68.40 | 52.10 | 38.68 | 28.69 | 82.65 | 16.37 | 23.13 | 50.94 |
| CapDec | 68.31 | 50.64 | 36.86 | 26.78 | 92.37 | 18.42 | 25.20 | 51.29 |
| BLIP | 70.65 | 56.72 | 44.45 | 34.36 | 107.67 | 20.40 | 26.06 | 53.66 |
| ViT-GPT2 | 77.73 | 61.80 | 47.06 | 35.39 | 119.06 | 21.15 | 27.94 | 57.11 |
| **Photon (Ours)** | **74.65** | **58.03** | **43.51** | **32.33** | **108.59** | **20.01** | **26.87** | **55.06** |

### Efficiency and Speed

*(Evaluated using a 15-token caption generation)*

| Model | Total Params | Total GFLOPs | GPU FPS | CPU FPS |
|---|---|---|---|---|
| ClipCap | 230M | 5.94 | 6.38 | 1.13 |
| SmallCap | 257.61M | 9.24 | 6.57 | 2.60 |
| BLIP | 224M | 132.21 | 3.77 | 0.51 |
| ViT-GPT2 | 239.20M | 17.33 | 5.28 | 1.17 |
| **Photon (Ours)** | **33.91M** | **3.72** | **9.09** | **4.21** |

---

## Reproducibility Note

*Please note: Earlier draft versions of this repository contained unscaled metrics (e.g., CIDEr 1.088) and component-isolated benchmarks (e.g., 0.49 GFLOPs and 942 FPS for the decoder alone). The values listed in this README have been synchronized to match the finalized, official publication figures.*

---

## License

This project is released under the MIT License. Portions of the codebase related to MobileCLIP are subject to Apple's respective licenses.

---

## Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{kodi2026photon,
  title={Photon: Efficient Prefix-Conditioned Image Captioning with Lightweight Transformer Decoding},
  author={Kodi, Lakshmi Ganapathi and Chauhan, Dharmendra and Gangireddy, Dhanush Reddy and Chaudhary, Sonam Kumari and Yeturu, Kalidas and Srivastava, Nikhil},
  journal={Machine Learning},
  volume={115},
  number={177},
  year={2026},
  publisher={Springer},
  doi={10.1007/s10994-026-07072-4}
}
```
