# mobilecap_modern_beam.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# ==========================================
# 1. MODERN COMPONENTS
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[2]
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class ModernAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len):
        super().__init__()
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_len)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(v, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

# ==========================================
# 2. MODEL
# ==========================================

class NanoLlamaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["d_model"])
        self.attn = ModernAttention(
            cfg["d_model"], cfg["num_heads"],
            cfg["max_length"] + cfg["num_img_tokens"]
        )
        self.ffn_norm = RMSNorm(cfg["d_model"])
        self.ffn = SwiGLUFFN(cfg["d_model"], cfg["ffn_hidden"])

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MobileCapModern(nn.Module):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.num_img_tokens = cfg["num_img_tokens"]

        self.proj = nn.Sequential(
            nn.Linear(cfg["img_emb_dim"], cfg["d_model"] * self.num_img_tokens),
            nn.GELU(),
            nn.Linear(cfg["d_model"] * self.num_img_tokens, cfg["d_model"] * self.num_img_tokens),
        )

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.layers = nn.ModuleList([NanoLlamaBlock(cfg) for _ in range(cfg["num_layers"])])
        self.norm = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, img_emb, input_ids):
        B, T = input_ids.shape
        img_tokens = self.proj(img_emb).view(B, self.num_img_tokens, self.cfg["d_model"])
        text_emb = self.token_embedding(input_ids)
        x = torch.cat([img_tokens, text_emb], dim=1)

        seq_len = x.shape[1]
        mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()
        mask[:, :self.num_img_tokens] = True

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.lm_head(x[:, self.num_img_tokens:, :])

    # =====================================================
    # GENERATION (GREEDY + BEAM SEARCH)
    # =====================================================
    def generate(
        self,
        img_emb,
        tokenizer,
        max_new_tokens=20,
        temperature=0.7,
        beam_size=1,
    ):
        self.eval()
        device = img_emb.device

        BOS = tokenizer.cls_token_id
        EOS = tokenizer.sep_token_id

        # ---------- GREEDY ----------
        if beam_size == 1:
            input_ids = torch.tensor([[BOS]], device=device)
            for _ in range(max_new_tokens):
                logits = self.forward(img_emb, input_ids)
                next_token = torch.argmax(logits[:, -1, :] / temperature, dim=-1).unsqueeze(1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == EOS:
                    break
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # ---------- BEAM SEARCH ----------
        beams = [(torch.tensor([[BOS]], device=device), 0.0)]

        for _ in range(max_new_tokens):
            new_beams = []
            for tokens, score in beams:
                if tokens[0, -1].item() == EOS:
                    new_beams.append((tokens, score))
                    continue

                logits = self.forward(img_emb, tokens)
                log_probs = torch.log_softmax(logits[:, -1, :] / temperature, dim=-1)
                topk_logp, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                for k in range(beam_size):
                    nt = topk_ids[:, k].unsqueeze(1)
                    ns = score + topk_logp[:, k].item()
                    new_beams.append((torch.cat([tokens, nt], dim=1), ns))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(b[0][0, -1].item() == EOS for b in beams):
                break

        return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)

# ==========================================
# 3. BUILDER
# ==========================================

CFG_MODERN = {
    "d_model": 256,
    "num_heads": 8,
    "ffn_hidden": 768,
    "num_layers": 6,
    "vocab_size": 8000,
    "img_emb_dim": 512,
    "max_length": 32,
    "num_img_tokens": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def build_modern_model(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
    })

    CFG_MODERN["vocab_size"] = len(tokenizer)
    model = MobileCapModern(tokenizer, CFG_MODERN).to(CFG_MODERN["device"])
    model.apply(_init_weights)
    return model, tokenizer

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
