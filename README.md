# ⚡ Photon-12M: Extreme High-Efficiency Image Captioning

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

**Photon-12M** is a nano-scale Vision–Language Model (VLM) designed for **extreme efficiency** in real-time and edge-constrained environments (Robotics, IoT, Wearables, Smart Cameras).

By bridging a **frozen MobileCLIP-S1 vision encoder** with a custom **12M-parameter Nano-LLaMA decoder**, Photon-12M achieves near-SOTA captioning quality for its size class while delivering **ultra-high throughput (942 FPS)** on consumer GPUs.

---

## 📄 Abstract

Modern multimodal models such as LLaVA-7B or BLIP-2 achieve strong language understanding but incur massive computational costs (≈140 GFLOPs per caption), making them unsuitable for real-time or embedded systems.

We introduce **Photon-12M**, a strictly efficient architecture operating at **0.49 GFLOPs**. A novel **1-to-8 Latent Expansion Bridge** converts a single global visual embedding into a compact multi-token context, enabling fine-grained caption generation without spatial grid attention. On COCO Val2017, Photon-12M achieves a **CIDEr score of 1.088**, matching models **100× larger**, while running **600× faster**.

---

## 🧠 Methodology & Architecture

Photon-12M consists of three carefully optimized components.

### 1️⃣ Vision Encoder — MobileCLIP-S1

* **Backbone:** MobileCLIP-S1 (mobile-optimized ViT-B/16 variant)
* **Strategy:** Frozen feature extraction
* **Output:** Single 512-D global image embedding (L2-normalized)

This ensures strong visual semantics with minimal compute overhead.

---

### 2️⃣ 1-to-8 Latent Expansion Bridge

Tiny decoders often suffer from *context collapse* when conditioned on a single token.

* **Our solution:** A lightweight MLP projects a `(1 × 512)` image vector into **8 latent tokens of size 256**
* **Effect:** Acts as a compact working memory, allowing separation of objects, attributes, and actions

This dramatically improves caption diversity and detail at negligible cost.

---

### 3️⃣ Nano-LLaMA Decoder (12M Parameters)

A miniaturized transformer inspired by LLaMA, optimized for small-scale deployment:

* **RoPE (Rotary Positional Embeddings)** – robust relative positioning
* **SwiGLU activations** – higher representational capacity
* **RMSNorm (pre-norm)** – stable FP16 training

Despite its size, the decoder exhibits strong compositional language ability.

---

## 📊 Quantitative Results

All benchmarks were conducted on an **NVIDIA RTX 4060 Laptop GPU (8GB VRAM)**.

### ⚡ Computational Efficiency

| Model          | Params  | FPS       | Latency     | GFLOPs   |
| -------------- | ------- | --------- | ----------- | -------- |
| LLaVA-1.5 (7B) | 7,000M  | ~1.5      | ~600 ms     | ~140.0   |
| BLIP-Base      | 224M    | ~45       | ~22 ms      | ~15.0    |
| **Photon-12M** | **12M** | **942.3** | **1.06 ms** | **0.49** |

---

### 🖼 Caption Quality (COCO Val2017)

| Metric    | Score     |
| --------- | --------- |
| **CIDEr** | **1.088** |
| BLEU-4    | 0.327     |
| ROUGE-L   | 0.552     |
| METEOR    | 0.269     |

CIDEr > 1.0 indicates strong semantic agreement with human captions.

---

## 📦 Installation & Setup (Docker — Recommended)

Docker is the **official and recommended** way to run Photon-12M. It guarantees reproducibility across systems and handles all native dependencies (CUDA, OpenCV, Java for METEOR, etc.).

### 🔨 Build the Docker image

```bash
docker build -t mobilecap .
```

### ▶ Run the container (GPU)

```bash
docker run --gpus all -it -v $(pwd):/workspace mobilecap
```

> ⚠️ Ensure `nvidia-container-toolkit` is installed on your host.

The MobileCLIP model weights (`models/mobileclip_s1.pt`) are automatically downloaded during the Docker build.

---

## 🚀 Inference & Demos

### 1️⃣ Live Video Captioning (Webcam / RTSP)

Run the Flask-based streaming demo:

```bash
python benchmarks/live_video_test.py
```

#### 🔧 RTSP Stream Configuration

To use an RTSP camera, edit **line 14** in:

```text
benchmarks/stream.py
```

```python
STREAM_URL = "rtsp://admin:admin@123@10.23.8.100:554/stream"
```

Then start the server:

```bash
python stream.py --host 0.0.0.0 --port 5000
```

Open your browser at:

```text
http://localhost:5000
```

---

### 2️⃣ Raw Tokenizer & Throughput Test

To test tokenizer speed and raw inference throughput:

```bash
python benchmarks/speedtest.py
```

---

## 📂 Repository Structure

```text
Photon-12M/
├── architecture/            # Core model components
├── training/                # Training & RL pipelines
├── tokenizer/               # Custom BPE tokenizer (8k vocab)
├── benchmarks/              # Speed, quality & streaming demos
├── models/                  # Downloaded MobileCLIP weights
├── train_nano.py            # Nano-LLaMA training script
├── mobilecap_modern.py      # Model architecture definition
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📜 Citation

```bibtex
@misc{photon12m,
  title        = {Photon-12M: Extreme High-Efficiency Image Captioning via Nano-LLaMA},
  author       = {Ganapathi},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\\url{https://github.com/ganapathi1578/Photon-12M}}
}
```

---

## 📄 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

> **Photon-12M demonstrates that extreme efficiency and strong multimodal reasoning are not mutually exclus
