"""
Transformer Model Architecture for Web Development LLM
Implements a modern LLM architecture with RMSNorm, RoPE, SwiGLU, and KV Cache
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: (batch, seq_len, n_heads, head_dim)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
    # Match seq_len
    freqs_cis = freqs_cis[:, :xq_.shape[1], :, :]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and KV Cache"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        Q, K = apply_rotary_emb(Q, K, freqs_cis)
        
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=1)
            V = torch.cat([V_cache, V], dim=1)
            
        new_kv_cache = (K, V)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores + mask
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(attn_output)
        return output, new_kv_cache


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # In SwiGLU, parameter count is kept similar by adjusting hidden_dim
        hidden_dim = int(2 * d_ff / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv_cache = self.attention(self.ln1(x), freqs_cis, mask, kv_cache)
        x = x + self.dropout(h)
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x, new_kv_cache


class WebDevLLM(nn.Module):
    """
    Modern Web Development Language Model
    Features RMSNorm, RoPE, SwiGLU, and KV Caching
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Precompute RoPE frequencies for up to 2x max length
        # Using register_buffer makes it part of state dict and properly device-managed
        self.register_buffer("freqs_cis_real", torch.view_as_real(precompute_freqs_cis(d_model // n_heads, max_seq_length * 2)))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_causal_mask(self, seq_len: int, past_len: int, device: torch.device) -> Optional[torch.Tensor]:
        total_len = past_len + seq_len
        if seq_len > 1:
            mask = torch.full((seq_len, total_len), float('-inf'), device=device)
            mask = torch.triu(mask, diagonal=past_len + 1)
            return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_len)
        return None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Convert real buffer back to complex
        freqs_cis = torch.view_as_complex(self.freqs_cis_real)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        
        past_len = 0 if kv_caches is None else kv_caches[0][0].shape[1]
        
        mask = self.get_causal_mask(seq_len, past_len, device)
        
        if attention_mask is not None:
            attn_mask_float = torch.zeros_like(attention_mask, dtype=torch.float, device=device)
            attn_mask_float = attn_mask_float.masked_fill(attention_mask == 0, float('-inf'))
            attn_mask_float = attn_mask_float.unsqueeze(1).unsqueeze(2)
            if mask is None:
                mask = attn_mask_float
            else:
                mask = mask + attn_mask_float
                
        new_kv_caches = []
        for i, block in enumerate(self.transformer_blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, freqs_cis, mask, kv)
            if use_cache:
                new_kv_caches.append(new_kv)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_kv_caches
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        
        kv_caches = None
        current_input = input_ids
        start_pos = 0
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, kv_caches = self.forward(
                    current_input, 
                    kv_caches=kv_caches, 
                    start_pos=start_pos, 
                    use_cache=True
                )
                
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
                
                current_input = next_token
                start_pos += logits.shape[1]
                
        return generated
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import ModelConfig
    
    model = WebDevLLM(
        vocab_size=ModelConfig.vocab_size,
        d_model=ModelConfig.d_model,
        n_layers=ModelConfig.n_layers,
        n_heads=ModelConfig.n_heads,
        d_ff=ModelConfig.d_ff,
        max_seq_length=ModelConfig.max_seq_length,
        dropout=ModelConfig.dropout,
    )
    
    print("Model created successfully with modern architecture!")
    print(f"Total parameters: {model.count_parameters():,}")
    
    batch_size = 2
    seq_len = 128
    dummy_input = torch.randint(0, ModelConfig.vocab_size, (batch_size, seq_len))
    
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    
    print("Testing generate with KV cache...")
    gen = model.generate(torch.randint(0, ModelConfig.vocab_size, (batch_size, 5)), max_length=10)
    print(f"Generated shape: {gen.shape}")
    print("Forward pass and generation successful!")
