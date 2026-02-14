"""
Unit tests for the tokenizer
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import WebDevTokenizer


def test_tokenizer_initialization():
    """Test tokenizer initialization"""
    tokenizer = WebDevTokenizer(vocab_size=1000)
    
    assert tokenizer.vocab_size == 1000
    assert len(tokenizer.special_tokens) > 0
    assert "pad_token" in tokenizer.special_tokens


def test_tokenizer_training():
    """Test tokenizer training"""
    texts = [
        "function test() { return true; }",
        "const App = () => { return <div>Hello</div>; }",
        ".container { display: flex; }",
    ]
    
    tokenizer = WebDevTokenizer(vocab_size=500)
    tokenizer.train(texts, verbose=False)
    
    assert len(tokenizer.vocab) > 0
    assert len(tokenizer.inverse_vocab) > 0


def test_encode_decode():
    """Test encoding and decoding"""
    texts = [
        "function test() { return true; }",
        "const App = () => <div>Hello</div>;",
    ]
    
    tokenizer = WebDevTokenizer(vocab_size=500)
    tokenizer.train(texts, verbose=False)
    
    test_text = "function test() { return true; }"
    
    # Encode
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    
    # Decode
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert isinstance(decoded, str)


def test_special_tokens():
    """Test special tokens are in vocabulary"""
    tokenizer = WebDevTokenizer(vocab_size=500)
    texts = ["test"]
    tokenizer.train(texts, verbose=False)
    
    for token_name, token in tokenizer.special_tokens.items():
        assert token in tokenizer.vocab
        assert token_name in tokenizer.special_token_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
