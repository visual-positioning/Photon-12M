# mobilecap_modern.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# ==========================================
# 1. MODERN COMPONENTS (The "LlaMA" Blocks)
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Faster and more stable than LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embeddings).
    Allows the model to understand relative positions better than absolute ones.
    """
    def __init__(self, dim, max_seq_len=128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Create cos/sin cache [Seq_Len, Dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[2]
        # Dynamically resize if needed (simple implementation)
        if seq_len > self.cos_cached.shape[2]:
             return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Ensure shapes match for broadcasting
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Gated mechanism that performs better than standard ReLU MLPs.
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False) # Value
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False) # Output
        
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
        
        # 1. Projections [B, T, D] -> [B, H, T, HeadDim]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply RoPE
        cos, sin = self.rope(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 3. Flash Attention (Scaled Dot Product)
        # dropout_p=0 during inference
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0 if not self.training else 0.1)
        
        # 4. Recombine
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

# ==========================================
# 2. THE MODEL (Nano-LlaMA)
# ==========================================

class NanoLlamaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["d_model"])
        self.attn = ModernAttention(cfg["d_model"], cfg["num_heads"], cfg["max_length"] + cfg["num_img_tokens"])
        
        self.ffn_norm = RMSNorm(cfg["d_model"])
        self.ffn = SwiGLUFFN(cfg["d_model"], cfg["ffn_hidden"])

    def forward(self, x, mask=None):
        # Pre-Norm Connection
        h = x + self.attn(self.attn_norm(x), mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class MobileCapModern(nn.Module):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.num_img_tokens = cfg["num_img_tokens"]
        
        # A. Projector: Expands 1 image vector -> 8 tokens
        # This is the "Bridge" between MobileCLIP and LLaMA
        self.proj = nn.Sequential(
            nn.Linear(cfg["img_emb_dim"], cfg["d_model"] * self.num_img_tokens),
            nn.GELU(),
            nn.Linear(cfg["d_model"] * self.num_img_tokens, cfg["d_model"] * self.num_img_tokens)
        )
        
        # B. Embeddings
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        # No pos_embedding here because we use RoPE inside the blocks!
        
        # C. Decoder Layers
        self.layers = nn.ModuleList([NanoLlamaBlock(cfg) for _ in range(cfg["num_layers"])])
        self.norm = RMSNorm(cfg["d_model"])
        
        # D. Head
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.token_embedding.weight = self.lm_head.weight # Weight tying

    def forward(self, img_emb, input_ids):
        B, T = input_ids.shape
        
        # 1. Prepare Inputs
        img_tokens = self.proj(img_emb).view(B, self.num_img_tokens, self.cfg["d_model"])
        text_emb = self.token_embedding(input_ids)
        x = torch.cat([img_tokens, text_emb], dim=1) # [B, 8 + T, D]
        
        # 2. Create Causal Mask
        seq_len = x.shape[1]
        # Standard causal mask (tril)
        mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()
        # Allow text to see ALL image tokens (since image tokens are at the start 0-7)
        mask[:, :self.num_img_tokens] = True 
        
        # 3. Forward Pass
        # We need to reshape mask for SDPA: [B, 1, T, T] or [T, T]
        # F.scaled_dot_product_attention handles [T, T] boolean masks automatically
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.norm(x)
        
        # 4. Extract Text Logits
        # We discard the first 8 tokens (image tokens) from the output
        text_out = x[:, self.num_img_tokens:, :]
        return self.lm_head(text_out)
        
    def generate(self, img_emb, tokenizer, max_new_tokens=20, temperature=0.7):
        """Batch-Safe generation loop"""
        self.eval()
        B = img_emb.shape[0] # Get batch size
        device = img_emb.device
        
        # Start with [CLS]
        # Ensure we replicate the start token for the whole batch
        start_token = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
        input_ids = torch.tensor([[start_token]] * B, device=device)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(img_emb, input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # --- FIX FOR BATCH CRASH ---
                # Only check for early stopping if we are processing a SINGLE image.
                # If we are benchmarking a batch (B > 1), we force full length generation
                # to ensure consistent speed measurement.
                if B == 1:
                     if next_token.item() == tokenizer.sep_token_id:
                        break
        
        # Return the first caption (for simple usage) or list of captions (if you expanded this)
        # For the speed test, we just return the first one to keep it simple.
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# ==========================================
# 3. BUILDER FUNCTION
# ==========================================

CFG_MODERN = {
    "d_model": 256,
    "num_heads": 8,            # 32 dim per head
    "ffn_hidden": 768,         # 3x d_model
    "num_layers": 6,           # Depth
    "vocab_size": 8000,        # Placeholder, updated on load
    "img_emb_dim": 512,        # MobileCLIP output
    "max_length": 32,
    "num_img_tokens": 8,       # Expand image to 8 tokens
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def build_modern_model(tokenizer_path="../tokenizer/mobilecap_tokenizer.json"):
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        return None, None
        
    tokenizer.add_special_tokens({
        "unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]", 
        "pad_token": "[PAD]", "mask_token": "[MASK]"
    })
    
    CFG_MODERN["vocab_size"] = len(tokenizer)
    
    model = MobileCapModern(tokenizer, CFG_MODERN).to(CFG_MODERN["device"])
    
    # Init weights (LlaMA style)
    model.apply(_init_weights)
    
    return model, tokenizer

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)