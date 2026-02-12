"""
Custom Tokenizer for Web Development LLM
Implements BPE (Byte Pair Encoding) tokenization with web development specific vocabulary
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import pickle


class WebDevTokenizer:
    """
    Custom tokenizer optimized for web development content
    Implements Byte Pair Encoding (BPE) algorithm
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = special_tokens or {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "code_start": "<CODE>",
            "code_end": "</CODE>",
            "html_tag": "<HTML>",
            "css_tag": "<CSS>",
            "js_tag": "<JS>",
        }
        
        # Vocabulary and merges
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        
        # Token IDs for special tokens
        self.special_token_ids: Dict[str, int] = {}
        
        # Web development specific patterns
        self.code_patterns = {
            'html_tag': r'<[^>]+>',
            'css_property': r'[a-z-]+\s*:',
            'js_function': r'function\s+\w+',
            'class_name': r'class\s+\w+',
            'variable': r'(var|let|const)\s+\w+',
            'import': r'import\s+.*from',
        }
        
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Get frequency of adjacent pairs in vocabulary"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """Merge the most frequent pair in vocabulary"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization"""
        # Preserve code blocks
        text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace(' ', '▁'), text)
        
        # Preserve HTML tags
        text = re.sub(r'<[^>]+>', lambda m: m.group(0).replace(' ', '▁'), text)
        
        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        Train the tokenizer on a corpus of texts
        
        Args:
            texts: List of training texts
            verbose: Print training progress
        """
        if verbose:
            print("Training tokenizer...")
            print(f"Target vocabulary size: {self.vocab_size}")
        
        # Initialize vocabulary with special tokens
        current_vocab_size = 0
        for token_name, token in self.special_tokens.items():
            self.vocab[token] = current_vocab_size
            self.special_token_ids[token_name] = current_vocab_size
            current_vocab_size += 1
        
        # Preprocess and split texts into words
        word_freqs = Counter()
        for text in texts:
            text = self._preprocess_text(text)
            words = text.split()
            word_freqs.update(words)
        
        # Convert words to character sequences
        word_freqs = {
            tuple(word) + ('</w>',): freq
            for word, freq in word_freqs.items()
        }
        
        # Get all unique characters
        chars = set()
        for word in word_freqs.keys():
            chars.update(word)
        
        # Add characters to vocabulary
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = current_vocab_size
                current_vocab_size += 1
        
        # Perform BPE merges
        merge_count = 0
        while current_vocab_size < self.vocab_size:
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = current_vocab_size
                self.merges[best_pair] = merge_count
                current_vocab_size += 1
                merge_count += 1
            
            if verbose and merge_count % 1000 == 0:
                print(f"Merges: {merge_count}, Vocab size: {current_vocab_size}")
        
        # Create inverse vocabulary
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        if verbose:
            print(f"\nTokenizer training complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE"""
        if not word:
            return []
        
        # Start with characters
        word = tuple(word) + ('</w>',)
        
        # Apply merges
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            
            # Find the pair with the lowest merge index
            valid_pairs = [(pair, self.merges.get(pair, float('inf'))) for pair in pairs]
            if not any(idx != float('inf') for _, idx in valid_pairs):
                break
            
            best_pair = min(valid_pairs, key=lambda x: x[1])[0]
            if best_pair not in self.merges:
                break
            
            # Merge the pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(''.join(best_pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        
        return list(word)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Add BOS and EOS tokens
        Returns:
            List of token IDs
        """
        text = self._preprocess_text(text)
        words = text.split()
        
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        # Convert tokens to IDs
        token_ids = [
            self.vocab.get(token, self.special_token_ids['unk_token'])
            for token in tokens
        ]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = (
                [self.special_token_ids['bos_token']] +
                token_ids +
                [self.special_token_ids['eos_token']]
            )
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        Returns:
            Decoded text
        """
        special_ids = set(self.special_token_ids.values())
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.inverse_vocab.get(token_id, self.special_tokens['unk_token']))
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = text.replace('▁', ' ')
        text = text.strip()
        
        return text
    
    def save(self, save_path: Path):
        """Save tokenizer to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(save_path / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(save_path / 'merges.pkl', 'wb') as f:
            pickle.dump(self.merges, f)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'special_token_ids': self.special_token_ids,
        }
        with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'WebDevTokenizer':
        """Load tokenizer from disk"""
        load_path = Path(load_path)
        
        # Load config
        with open(load_path / 'config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=config['vocab_size'],
            special_tokens=config['special_tokens']
        )
        
        # Load vocabulary
        with open(load_path / 'vocab.json', 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)
        
        # Load merges
        with open(load_path / 'merges.pkl', 'rb') as f:
            tokenizer.merges = pickle.load(f)
        
        # Restore other attributes
        tokenizer.special_token_ids = config['special_token_ids']
        tokenizer.inverse_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
        
        print(f"Tokenizer loaded from {load_path}")
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)


if __name__ == "__main__":
    # Test the tokenizer
    sample_texts = [
        "function HelloWorld() { return <div>Hello World</div>; }",
        "const App = () => { const [state, setState] = useState(0); }",
        ".container { display: flex; justify-content: center; }",
        "import React from 'react'; export default App;",
    ]
    
    tokenizer = WebDevTokenizer(vocab_size=1000)
    tokenizer.train(sample_texts, verbose=True)
    
    # Test encoding/decoding
    test_text = "function test() { return <div>Test</div>; }"
    print(f"\nOriginal text: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
