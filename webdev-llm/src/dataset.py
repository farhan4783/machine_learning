"""
Dataset class for Web Development LLM training
Handles data loading, preprocessing, and batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import random
from tqdm import tqdm


class WebDevDataset(Dataset):
    """
    PyTorch Dataset for web development text data
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
    ):
        """
        Args:
            data_path: Path to processed data file
            tokenizer: WebDevTokenizer instance
            max_length: Maximum sequence length
            stride: Stride for creating overlapping sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Load data
        self.samples = self._load_data(data_path)
        
    def _load_data(self, data_path: Path) -> List[List[int]]:
        """Load and tokenize data"""
        print(f"Loading data from {data_path}...")
        
        samples = []
        
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in tqdm(data, desc="Tokenizing"):
                text = item.get('text', '')
                if not text:
                    continue
                
                # Encode text
                token_ids = self.tokenizer.encode(text, add_special_tokens=True)
                
                # Create overlapping chunks
                for i in range(0, len(token_ids), self.stride):
                    chunk = token_ids[i:i + self.max_length]
                    if len(chunk) >= 50:  # Minimum chunk size
                        samples.append(chunk)
        
        elif data_path.suffix == '.txt':
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Encode entire text
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Create overlapping chunks
            for i in range(0, len(token_ids), self.stride):
                chunk = token_ids[i:i + self.max_length]
                if len(chunk) >= 50:
                    # Add special tokens
                    chunk = (
                        [self.tokenizer.special_token_ids['bos_token']] +
                        chunk +
                        [self.tokenizer.special_token_ids['eos_token']]
                    )
                    samples.append(chunk)
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Target token IDs (shifted input_ids)
        """
        token_ids = self.samples[idx]
        
        # Pad if necessary
        if len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.special_token_ids['pad_token']] * padding_length
        else:
            token_ids = token_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [
            1 if token_id != self.tokenizer.special_token_ids['pad_token'] else 0
            for token_id in token_ids
        ]
        
        # For language modeling, labels are the same as input_ids
        # The model will learn to predict the next token
        labels = token_ids.copy()
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class WebDevDataModule:
    """
    Data module for managing train/val/test datasets and dataloaders
    """
    
    def __init__(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Optional[Path],
        tokenizer,
        batch_size: int = 32,
        max_length: int = 1024,
        num_workers: int = 4,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        print("Setting up datasets...")
        
        self.train_dataset = WebDevDataset(
            self.train_path,
            self.tokenizer,
            self.max_length
        )
        
        self.val_dataset = WebDevDataset(
            self.val_path,
            self.tokenizer,
            self.max_length
        )
        
        if self.test_path:
            self.test_dataset = WebDevDataset(
                self.test_path,
                self.tokenizer,
                self.max_length
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


if __name__ == "__main__":
    # Test the dataset
    from tokenizer import WebDevTokenizer
    from config import DataConfig
    
    # Create dummy data for testing
    dummy_data = [
        {"text": "function test() { return <div>Hello</div>; }"},
        {"text": "const App = () => { return <h1>React App</h1>; }"},
        {"text": ".container { display: flex; align-items: center; }"},
    ]
    
    test_file = DataConfig.processed_data_dir / "test.json"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, 'w') as f:
        json.dump(dummy_data, f)
    
    # Create tokenizer
    tokenizer = WebDevTokenizer(vocab_size=1000)
    tokenizer.train([item['text'] for item in dummy_data])
    
    # Create dataset
    dataset = WebDevDataset(test_file, tokenizer, max_length=128)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
