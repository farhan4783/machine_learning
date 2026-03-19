"""
Data Preprocessing Script
Cleans and prepares raw data for training
"""

import json
from pathlib import Path
from typing import List, Dict
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataPreprocessor:
    """Preprocess raw web development data for training"""
    
    def __init__(self, raw_data_path: Path, output_dir: Path):
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep code-related ones
        # Keep: {}[]()<>.,;:!?'"=-+*/\|@#$%^&
        
        # Remove URLs (but keep them in code examples)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def clean_code(self, code: str) -> str:
        """Clean code examples"""
        # Remove excessive blank lines
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        # Normalize indentation (convert tabs to spaces)
        code = code.replace('\t', '    ')
        
        return code.strip()
    
    def create_training_samples(self, data: List[Dict]) -> List[Dict]:
        """Create training samples from raw data"""
        samples = []
        
        for item in tqdm(data, desc="Creating samples"):
            text = item.get('text', '')
            code_examples = item.get('code_examples', [])
            topic = item.get('topic', 'General')
            
            # Clean text
            if text:
                cleaned_text = self.clean_text(text)
                
                if len(cleaned_text) >= 50:  # Minimum length
                    samples.append({
                        'text': cleaned_text,
                        'topic': topic,
                        'type': 'documentation',
                        'source': item.get('source', 'Unknown')
                    })
            
            # Process code examples
            for code in code_examples:
                cleaned_code = self.clean_code(code)
                
                if len(cleaned_code) >= 20:  # Minimum code length
                    # Add context to code
                    code_with_context = f"Here is a {topic} code example:\n\n{cleaned_code}"
                    
                    samples.append({
                        'text': code_with_context,
                        'topic': topic,
                        'type': 'code',
                        'source': item.get('source', 'Unknown')
                    })
            
            # Combine text and code for richer samples
            if text and code_examples:
                combined = f"{cleaned_text}\n\nExample code:\n\n{self.clean_code(code_examples[0])}"
                
                samples.append({
                    'text': combined,
                    'topic': topic,
                    'type': 'combined',
                    'source': item.get('source', 'Unknown')
                })
        
        return samples
    
    def augment_data(self, samples: List[Dict]) -> List[Dict]:
        """Augment training data"""
        augmented = samples.copy()
        
        # Add variations of code examples
        for sample in samples:
            if sample['type'] == 'code':
                # Create Q&A style samples
                code = sample['text']
                topic = sample['topic']
                
                # Question format
                qa_sample = {
                    'text': f"User: How do I write {topic} code?\n\nAssistant: {code}",
                    'topic': topic,
                    'type': 'qa',
                    'source': sample['source']
                }
                augmented.append(qa_sample)
                
                # Explanation format
                explain_sample = {
                    'text': f"User: Can you explain this {topic} code?\n\n{code}\n\nAssistant: This code demonstrates key concepts of {topic}.",
                    'topic': topic,
                    'type': 'explanation',
                    'source': sample['source']
                }
                augmented.append(explain_sample)
        
        return augmented
    
    def split_data(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> tuple:
        """Split data into train/val/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # First split: train and temp (val + test)
        train_data, temp_data = train_test_split(
            samples,
            test_size=(1 - train_ratio),
            random_state=42
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=42
        )
        
        return train_data, val_data, test_data
    
    def save_splits(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Save train/val/test splits"""
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(split_data)} samples to {output_path}")
    
    def process(self):
        """Run the full preprocessing pipeline"""
        print("Loading raw data...")
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"Loaded {len(raw_data)} raw items")
        
        # Create training samples
        samples = self.create_training_samples(raw_data)
        print(f"Created {len(samples)} training samples")
        
        # Augment data
        augmented_samples = self.augment_data(samples)
        print(f"Augmented to {len(augmented_samples)} samples")
        
        # Split data
        train_data, val_data, test_data = self.split_data(augmented_samples)
        print(f"\nData split:")
        print(f"  Train: {len(train_data)}")
        print(f"  Val: {len(val_data)}")
        print(f"  Test: {len(test_data)}")
        
        # Save splits
        self.save_splits(train_data, val_data, test_data)
        
        # Print statistics
        self.print_statistics(train_data, val_data, test_data)
    
    def print_statistics(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Print dataset statistics"""
        print("\n=== Dataset Statistics ===")
        
        for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            total_chars = sum(len(item['text']) for item in split_data)
            avg_length = total_chars / len(split_data) if split_data else 0
            
            topics = {}
            types = {}
            for item in split_data:
                topics[item['topic']] = topics.get(item['topic'], 0) + 1
                types[item['type']] = types.get(item['type'], 0) + 1
            
            print(f"\n{split_name} Set:")
            print(f"  Samples: {len(split_data)}")
            print(f"  Total characters: {total_chars:,}")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Topics: {len(topics)}")
            print(f"  Types: {types}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from config import DataConfig
    
    raw_data_path = DataConfig.raw_data_dir / "raw_data.json"
    
    preprocessor = DataPreprocessor(raw_data_path, DataConfig.processed_data_dir)
    preprocessor.process()
