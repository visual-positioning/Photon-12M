# Architecture

Photon is a lightweight prefix-conditioned image captioning model consisting of three components:

1. **Frozen MobileCLIP-S1 Encoder**: Extracts a 512-dimensional global visual embedding from the input image. The FastViT backbone ensures efficient extraction.
2. **Prefix Projection Module**: A two-layer MLP with a GELU activation that transforms the L2-normalized 512-D visual embedding into $K = 8$ visual prefix tokens.
3. **Compact Decoder-only Transformer**: A 6-layer Nano-LLaMA style Transformer with a hidden dimension of 256. It processes the concatenated visual prefix tokens and text embeddings using causal self-attention.

## Design Choices

* **Prefix Conditioning**: Instead of heavy cross-attention, Photon conditions generation on early prefix tokens, efficiently guiding the autoregressive process.
* **RoPE & SwiGLU**: Modern components like Rotary Positional Embeddings and SwiGLU activations maximize representational capacity within a small parameter budget.
* **Weight Tying**: The language modeling head is tied to the token embedding matrix to reduce parameter count.
* **Unified Self-Attention**: Modalities interact via a single shared self-attention operation over concatenated tokens, removing dedicated cross-attention overhead.

Source: `paper.pdf` Section 3.1 & architecture/mobilecap_modern.py
