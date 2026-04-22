# mobilecap_ablation.py
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
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[2]
        if seq_len > self.cos_cached.shape[2]:
             return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
        
        cos, sin = self.rope(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class CrossAttention(nn.Module):
    """Simple Cross Attention for 'cross_attn' strategy"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, context):
        # x: [B, T, D] (Text)
        # context: [B, S, D] (Image Tokens)
        B, T, C = x.shape
        S = context.shape[1]
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # No RoPE usually for Cross Attn context, or only on Q. Standard is no RoPE.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

# ==========================================
# 2. THE MODEL
# ==========================================

class NanoLlamaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.use_cross_attn = (cfg.get("conditioning_strategy") == "cross_attn")
        
        self.attn_norm = RMSNorm(self.d_model)
        # Max len buffer
        max_total_len = cfg["max_length"] + cfg.get("num_img_tokens", 0)
        self.attn = ModernAttention(self.d_model, cfg["num_heads"], max_total_len)
        
        if self.use_cross_attn:
            self.cross_norm = RMSNorm(self.d_model)
            self.cross_attn = CrossAttention(self.d_model, cfg["num_heads"])
        
        self.ffn_norm = RMSNorm(self.d_model)
        self.ffn = SwiGLUFFN(self.d_model, cfg["ffn_hidden"])

    def forward(self, x, mask=None, context=None):
        # Self Attention
        h = x + self.attn(self.attn_norm(x), mask)
        
        # Cross Attention (Optional)
        if self.use_cross_attn and context is not None:
            h = h + self.cross_attn(self.cross_norm(h), context)
            
        # FFN
        out = h + self.ffn(self.ffn_norm(h))
        return out

class MobileCapModern(nn.Module):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.strategy = cfg.get("conditioning_strategy", "prefix")
        self.num_img_tokens = cfg.get("num_img_tokens", 8)
        
        # A. Projector
        if self.strategy == "mean_add":
            # Project to single vector of d_model size
            self.proj = nn.Sequential(
                nn.Linear(cfg["img_emb_dim"], cfg["d_model"]),
                nn.GELU(),
                nn.Linear(cfg["d_model"], cfg["d_model"])
            )
        else:
            # Project to [num_img_tokens * d_model]
            self.proj = nn.Sequential(
                nn.Linear(cfg["img_emb_dim"], cfg["d_model"] * self.num_img_tokens),
                nn.GELU(),
                nn.Linear(cfg["d_model"] * self.num_img_tokens, cfg["d_model"] * self.num_img_tokens)
            )
        
        # B. Embeddings
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        
        # C. Decoder Layers
        self.layers = nn.ModuleList([NanoLlamaBlock(cfg) for _ in range(cfg["num_layers"])])
        self.norm = RMSNorm(cfg["d_model"])
        
        # D. Head
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, img_emb, input_ids):
        B, T = input_ids.shape
        text_emb = self.token_embedding(input_ids)
        
        # --- Handle Strategies ---
        x = None
        mask = None
        cross_context = None
        
        if self.strategy == "mean_add":
            # Add image vector to every text token
            img_vec = self.proj(img_emb).unsqueeze(1) # [B, 1, D]
            x = text_emb + img_vec # Broadcast add
            
            # Standard Mask
            mask = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
            
        elif self.strategy == "cross_attn":
            # Image tokens act as context for cross attention layers
            x = text_emb
            cross_context = self.proj(img_emb).view(B, self.num_img_tokens, self.cfg["d_model"])
            
            # Standard Mask (only for text self-attn)
            mask = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
            
        else: # "prefix" or "global_token"
            # Concatenate [ImgTokens, TextTokens]
            img_tokens = self.proj(img_emb).view(B, self.num_img_tokens, self.cfg["d_model"])
            x = torch.cat([img_tokens, text_emb], dim=1)
            
            # Mask allowing text to see image prefix
            seq_len = x.shape[1]
            mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()
            mask[:, :self.num_img_tokens] = True 

        # --- Forward Pass ---
        for layer in self.layers:
            x = layer(x, mask=mask, context=cross_context)
            
        x = self.norm(x)
        
        # --- Output extraction ---
        if self.strategy in ["mean_add", "cross_attn"]:
            # No prefix tokens to remove
            text_out = x
        else:
            # Remove prefix
            text_out = x[:, self.num_img_tokens:, :]
            
        return self.lm_head(text_out)
        
    def generate(self, img_emb, tokenizer, max_new_tokens=20, temperature=0.7):
        self.eval()
        B = img_emb.shape[0]
        device = img_emb.device
        input_ids = torch.tensor([[tokenizer.cls_token_id]] * B, device=device)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(img_emb, input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.sep_token_id: break
        
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Keep builder unchanged, but make sure it uses the class above
CFG_MODERN = {
    "d_model": 256, "num_heads": 8, "ffn_hidden": 768, "num_layers": 6,
    "vocab_size": 8000, "img_emb_dim": 512, "max_length": 32, "num_img_tokens": 8,
    "device": "cuda", "conditioning_strategy": "prefix"
}

def build_modern_model(tokenizer_path, override_cfg=None):
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    except:
        return None, None

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]"})

    final_cfg = CFG_MODERN.copy()
    if override_cfg: final_cfg.update(override_cfg)
    final_cfg["vocab_size"] = len(tokenizer)
    
    device = final_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = MobileCapModern(tokenizer, final_cfg).to(device)
    
    def _init_weights(m):
        if isinstance(m, nn.Linear): torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, std=0.02)
    
    model.apply(_init_weights)
    return model, tokenizer