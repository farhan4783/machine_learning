"""
Card Generation System
Generates educational knowledge cards using the trained model
"""

import torch
from typing import List, Dict, Optional
from pathlib import Path
import json

from model import WebDevLLM
from tokenizer import WebDevTokenizer
from config import ModelConfig, InferenceConfig, DataConfig
from ai_engine import WebDevAIEngine


class CardGenerator:
    """Generate educational cards about web development topics"""
    
    def __init__(
        self,
        model: Optional[WebDevLLM] = None,
        tokenizer: Optional[WebDevTokenizer] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()
        self.tokenizer = tokenizer
        self.ai_engine = WebDevAIEngine(neural_model=self.model, tokenizer=self.tokenizer, device=self.device)
    
    def generate_card(
        self,
        topic: str,
        card_type: str = 'concept',
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive knowledge card for a given topic
        """
        return self.ai_engine.generate_card(
            topic=topic,
            card_type=card_type,
            max_length=max_length,
            temperature=temperature
        )
    
    def _parse_card_content(self, text: str, topic: str, card_type: str) -> Dict[str, str]:
        """Parse generated text into structured card format"""
        # Simple parsing - can be enhanced with more sophisticated NLP
        lines = text.split('\n')
        
        card = {
            'topic': topic,
            'type': card_type,
            'title': f"{topic} - {card_type.replace('_', ' ').title()}",
            'content': text,
            'sections': []
        }
        
        # Extract code blocks if present
        code_blocks = []
        in_code = False
        current_code = []
        
        for line in lines:
            if '```' in line or 'function' in line or 'const' in line or 'class' in line:
                if not in_code:
                    in_code = True
                    current_code = [line]
                else:
                    current_code.append(line)
                    code_blocks.append('\n'.join(current_code))
                    in_code = False
                    current_code = []
            elif in_code:
                current_code.append(line)
        
        if code_blocks:
            card['code_examples'] = code_blocks
        
        return card
    
    def generate_batch(
        self,
        topics: List[str],
        card_type: str = 'concept',
        **kwargs
    ) -> List[Dict[str, str]]:
        """Generate multiple cards"""
        cards = []
        
        for topic in topics:
            card = self.generate_card(topic, card_type, **kwargs)
            cards.append(card)
        
        return cards
    
    def generate_comprehensive_card(self, topic: str) -> Dict[str, any]:
        """Generate a comprehensive card with multiple sections"""
        comprehensive_card = {
            'topic': topic,
            'title': f"Complete Guide to {topic}",
            'sections': {}
        }
        
        # Generate different types of content
        card_types = ['concept', 'code_example', 'best_practices', 'use_cases']
        
        for card_type in card_types:
            card = self.generate_card(topic, card_type, max_length=256)
            comprehensive_card['sections'][card_type] = card['content']
        
        return comprehensive_card
    
    def save_card(self, card: Dict, output_path: Path):
        """Save card to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
        
        print(f"Saved card to {output_path}")
    
    @classmethod
    def load_model(cls, checkpoint_path: Path, device: Optional[str] = None) -> 'CardGenerator':
        """Load model from checkpoint"""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_cfg = checkpoint.get('config', {}).get('model', {})
        
        # Load tokenizer
        tokenizer = WebDevTokenizer.load(DataConfig.tokenizer_dir)
        
        # Create model
        model = WebDevLLM(
            vocab_size=model_cfg.get('vocab_size', len(tokenizer)),
            d_model=model_cfg.get('d_model', ModelConfig.d_model),
            n_layers=model_cfg.get('n_layers', ModelConfig.n_layers),
            n_heads=model_cfg.get('n_heads', ModelConfig.n_heads),
            d_ff=model_cfg.get('d_ff', ModelConfig.d_ff),
            max_seq_length=model_cfg.get('max_seq_length', ModelConfig.max_seq_length),
            dropout=model_cfg.get('dropout', ModelConfig.dropout),
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return cls(model, tokenizer, device)


def main():
    """Test card generation"""
    # Load model
    checkpoint_path = DataConfig.checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    print("Loading model...")
    generator = CardGenerator.load_model(checkpoint_path)
    
    # Generate sample cards
    topics = [
        "React Hooks",
        "CSS Flexbox",
        "JavaScript Promises",
        "Node.js Express",
        "MongoDB Queries"
    ]
    
    print("\nGenerating cards...")
    for topic in topics:
        print(f"\n{'='*50}")
        print(f"Topic: {topic}")
        print('='*50)
        
        card = generator.generate_card(topic, card_type='concept')
        print(f"\n{card['content']}")
        
        # Save card
        output_path = DataConfig.base_dir / "generated_cards" / f"{topic.replace(' ', '_').lower()}.json"
        generator.save_card(card, output_path)


if __name__ == "__main__":
    main()
