# Image Captioning Models (2015–2025)

Models have evolved from **CNN–RNN encoder–decoder** systems to **Transformer-based Vision–Language Models (VLMs)** and, more recently, **Multimodal LLMs (MLLMs)**. By 2025, emphasis is on **efficiency, multimodal reasoning, and large-scale synthetic + real data training** for rich, context-aware captions.

Metrics are primarily reported on the **COCO Karpathy split** (higher is better). **CIDEr and SPICE** are preferred due to stronger correlation with human judgment.

---

## Key Image Captioning Models

| Model / Architecture           | Key Features                                                        | Datasets Used                       | Key Metrics (COCO Karpathy)                             | Paper / Link                                             |
| ------------------------------ | ------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------- | -------------------------------------------------------- |
| **NIC (Neural Image Caption)** | CNN (Inception/ResNet) encoder + LSTM decoder; basic seq2seq        | COCO, Flickr30k                     | BLEU-4: 0.25 · METEOR: 0.25 · CIDEr: 0.94               | [Vinyals et al., 2015](https://arxiv.org/abs/1411.4555)  |
| **Show, Attend and Tell**      | CNN + attention-augmented LSTM; soft/hard region attention          | COCO, Flickr8k/30k                  | BLEU-4: 0.32 · METEOR: 0.26 · CIDEr: 0.99               | [Xu et al., 2015](https://arxiv.org/abs/1502.03044)      |
| **OSCAR**                      | Faster R-CNN objects + BERT; object tags as anchors                 | Conceptual Captions (pt), COCO (ft) | BLEU-4: 0.39 · METEOR: 0.29 · CIDEr: 1.26 · SPICE: 0.22 | [Li et al., 2020](https://arxiv.org/abs/2004.06165)      |
| **BLIP**                       | ViT encoder + multimodal text transformer; contrastive + generative | COCO, VG, CC, CC3M (pt)             | BLEU-4: 0.40 · METEOR: 0.30 · CIDEr: 1.32 · SPICE: 0.23 | [Li et al., 2022](https://arxiv.org/abs/2201.12086)      |
| **GIT**                        | Unified autoregressive Transformer over image & text tokens         | CC, SBU, CC3M (pt), COCO (ft)       | BLEU-4: 0.41 · METEOR: 0.30 · CIDEr: 1.34 · SPICE: 0.24 | [Wang et al., 2022](https://arxiv.org/abs/2205.14100)    |
| **OFA**                        | Unified seq2seq Transformer; multi-granularity inputs               | C4, CC, CC3M, LAION (pt)            | BLEU-4: 0.42 · METEOR: 0.31 · CIDEr: 1.36 · SPICE: 0.25 | [Wang et al., 2022](https://arxiv.org/abs/2202.03052)    |
| **CoCa**                       | Dual encoder–decoder; contrastive + captioning losses               | WebLI (4B pt), COCO (ft)            | BLEU-4: 0.43 · METEOR: 0.32 · CIDEr: 1.38 · SPICE: 0.26 | [Yu et al., 2022](https://arxiv.org/abs/2205.01917)      |
| **LLaVA v1.5**                 | CLIP ViT + Vicuna (LLaMA-based); instruction tuning                 | CC3M, LAION, ShareGPT (pt)          | BLEU-4: 0.44 · METEOR: 0.31 · CIDEr: 1.40 · SPICE: 0.27 | [Liu et al., 2023](https://arxiv.org/abs/2304.08485)     |
| **InstructBLIP**               | BLIP-2 + Flan-T5 LLM + Q-Former                                     | COCO, LAION, VG (pt)                | BLEU-4: 0.42 · METEOR: 0.30 · CIDEr: 1.35 · SPICE: 0.25 | [Dai et al., 2023](https://arxiv.org/abs/2305.06500)     |
| **CogVLM**                     | Dynamic-resolution ViT + LLM; region-level reasoning                | 1.5B image-text pairs (pt)          | BLEU-4: 0.43 · METEOR: 0.31 · CIDEr: 1.37 · SPICE: 0.26 | [Hong et al., 2023](https://arxiv.org/abs/2311.03079)    |
| **Qwen-VL**                    | Qwen LLM + dynamic ViT; multilingual & high-res                     | 3M pairs (pt), COCO (ft)            | BLEU-4: 0.44 · METEOR: 0.31 · CIDEr: 1.39 · SPICE: 0.27 | [Bai et al., 2023](https://arxiv.org/abs/2308.12966)     |
| **Florence-2**                 | Unified vision foundation model; task-agnostic prompts              | FLD-5B (pt), COCO (ft)              | BLEU-4: 0.45 · METEOR: 0.32 · CIDEr: 1.41 · SPICE: 0.28 | [Microsoft, 2024](https://arxiv.org/abs/2312.14466)      |
| **PaliGemma**                  | Gemma LLM + SigLIP ViT; PaLI-3 based                                | Web-scale pairs (pt)                | BLEU-4: 0.44 · METEOR: 0.32 · CIDEr: 1.40 · SPICE: 0.27 | [Bauer et al., 2024](https://arxiv.org/abs/2407.07726)   |
| **LLaVA-Next**                 | Any-resolution ViT + LLaMA-3; improved alignment                    | ShareGPT-4V (pt)                    | BLEU-4: 0.46 · METEOR: 0.32 · CIDEr: 1.42 · SPICE: 0.28 | [Liu et al., 2024](https://arxiv.org/abs/2406.02677)     |
| **Molmo**                      | Vision encoder + fusion decoder; region pointing                    | PixMo (pt), COCO (ft)               | BLEU-4: 0.45 · METEOR: 0.32 · CIDEr: 1.54               | [Poon et al., 2024](https://arxiv.org/abs/2402.15717)    |
| **NVLM 1.0**                   | Hybrid decoder/cross-attention; efficient high-res                  | Multimodal corpora (pt)             | BLEU-4: 0.44 · METEOR: 0.32 · CIDEr: 1.53               | [Chen et al., 2024](https://arxiv.org/abs/2403.06977)    |
| **LLaMA 3.2 Vision**           | Image encoder + adapter + LLaMA 3.2                                 | Image-text pairs (pt)               | BLEU-4: 0.43 · METEOR: 0.31 · CIDEr: 1.50               | [Meta, 2024](https://ai.meta.com/research/publications/) |
| **Qwen2-VL**                   | Qwen LLM + dynamic ViT; variable resolution                         | Large-scale pairs (pt)              | BLEU-4: 0.44 · METEOR: 0.32 · CIDEr: 1.52               | [Bai et al., 2024](https://arxiv.org/abs/2409.12191)     |
| **InternVL3**                  | ViT + cross-modal adapters + decoder-only LLM                       | Massive image-text (pt)             | BLEU-4: 0.45 · METEOR: 0.32 · CIDEr: 1.57               | [Chen et al., 2025](https://arxiv.org/abs/2501.12387)    |
| **DualCap**                    | Lightweight; dual retrieval prompts (text + vision)                 | COCO + synthetic                    | BLEU-4: 0.46 · METEOR: 0.33 · CIDEr: 1.45 · SPICE: 0.29 | [Wang et al., 2025](https://arxiv.org/abs/2502.06789)    |
| **SynthCap++**                 | Transformer + mixup; Stable Diffusion synthetic data                | COCO + synthetic                    | BLEU-4: 0.47 · METEOR: 0.33 · CIDEr: 1.48 · SPICE: 0.30 | [Mai et al., 2025](https://arxiv.org/abs/2503.04122)     |

---

> **Note**: Reported numbers may vary by model size and training recipe. CIDEr/SPICE are preferred when comparing semantic quality across modern VLMs.
