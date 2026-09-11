"""
Unit tests for the model
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import WebDevLLM, MultiHeadAttention, TransformerBlock, precompute_freqs_cis
from config import ModelConfig


def test_model_initialization():
    """Test model can be initialized"""
    model = WebDevLLM(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_length=512,
    )
    
    assert model is not None
    assert model.count_parameters() > 0


def test_forward_pass():
    """Test forward pass"""
    model = WebDevLLM(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
    )
    
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, 1000)


def test_generation():
    """Test text generation"""
    model = WebDevLLM(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
    )
    
    input_ids = torch.randint(0, 1000, (1, 10))
    
    generated = model.generate(
        input_ids,
        max_length=20,
        temperature=1.0,
    )
    
    assert generated.shape[1] > input_ids.shape[1]


def test_attention_mechanism():
    """Test multi-head attention"""
    d_model = 256
    n_heads = 4
    attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model)
    freqs_cis = precompute_freqs_cis(dim=d_model // n_heads, end=seq_len)
    
    output, new_kv = attention(x, freqs_cis)
    
    assert output.shape == x.shape
    assert new_kv[0].shape[1] == seq_len


def test_transformer_block():
    """Test transformer block"""
    d_model = 256
    n_heads = 4
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=1024)
    
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model)
    freqs_cis = precompute_freqs_cis(dim=d_model // n_heads, end=seq_len)
    
    output, new_kv = block(x, freqs_cis)
    
    assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
