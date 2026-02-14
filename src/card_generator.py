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


class CardGenerator:
    """Generate educational cards about web development topics"""
    
    def __init__(
        self,
        model: WebDevLLM,
        tokenizer: WebDevTokenizer,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # Card templates
        self.templates = {
            'concept': "Explain the concept of {topic} in web development. Include key points and examples.",
            'code_example': "Provide a detailed code example for {topic}. Include comments and best practices.",
            'tutorial': "Create a step-by-step tutorial for {topic}. Make it beginner-friendly.",
            'comparison': "Compare and contrast {topic} with similar technologies. Highlight pros and cons.",
            'best_practices': "List the best practices for {topic}. Include common pitfalls to avoid.",
            'use_cases': "Describe real-world use cases for {topic}. Provide practical examples.",
        }
    
    def generate_card(
        self,
        topic: str,
        card_type: str = 'concept',
        max_length: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Dict[str, str]:
        """
        Generate a knowledge card for a given topic
        
        Args:
            topic: Web development topic
            card_type: Type of card (concept, code_example, tutorial, etc.)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            Dictionary containing card data
        """
        # Get template
        template = self.templates.get(card_type, self.templates['concept'])
        prompt = template.format(topic=topic)
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.special_token_ids['eos_token']
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        
        # Parse card content
        card = self._parse_card_content(generated_text, topic, card_type)
        
        return card
    
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
    def load_model(cls, checkpoint_path: Path, device: str = "cuda") -> 'CardGenerator':
        """Load model from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load tokenizer
        tokenizer = WebDevTokenizer.load(DataConfig.tokenizer_dir)
        
        # Create model
        model = WebDevLLM(
            vocab_size=len(tokenizer),
            d_model=ModelConfig.d_model,
            n_layers=ModelConfig.n_layers,
            n_heads=ModelConfig.n_heads,
            d_ff=ModelConfig.d_ff,
            max_seq_length=ModelConfig.max_seq_length,
            dropout=ModelConfig.dropout,
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
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
