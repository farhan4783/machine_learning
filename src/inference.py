"""
Inference utilities for the Web Development LLM
"""

import torch
from typing import List, Optional
from pathlib import Path

from model import WebDevLLM
from tokenizer import WebDevTokenizer
from config import ModelConfig, InferenceConfig, DataConfig


class InferenceEngine:
    """Inference engine for text generation"""
    
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
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated texts
        """
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Repeat for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate
        generated = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.special_token_ids['eos_token']
        )
        
        # Decode
        generated_texts = []
        for seq in generated:
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def complete_code(self, code_snippet: str, max_length: int = 256) -> str:
        """Complete a code snippet"""
        prompt = f"Complete this code:\n\n{code_snippet}"
        
        completions = self.generate(
            prompt,
            max_length=max_length,
            temperature=0.7,  # Lower temperature for code
            top_k=40,
        )
        
        return completions[0]
    
    def explain_code(self, code: str) -> str:
        """Explain what a code snippet does"""
        prompt = f"Explain this code:\n\n{code}\n\nExplanation:"
        
        explanations = self.generate(
            prompt,
            max_length=512,
            temperature=0.8,
        )
        
        return explanations[0]
    
    def answer_question(self, question: str) -> str:
        """Answer a web development question"""
        prompt = f"Question: {question}\n\nAnswer:"
        
        answers = self.generate(
            prompt,
            max_length=512,
            temperature=0.8,
        )
        
        return answers[0]
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: str = "cuda") -> 'InferenceEngine':
        """Create inference engine from checkpoint"""
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


if __name__ == "__main__":
    # Test inference
    checkpoint_path = DataConfig.checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please train the model first")
        exit(1)
    
    print("Loading model...")
    engine = InferenceEngine.from_checkpoint(checkpoint_path)
    
    # Test different inference modes
    print("\n=== Code Completion ===")
    code_snippet = "function fetchData() {"
    completion = engine.complete_code(code_snippet)
    print(f"Input: {code_snippet}")
    print(f"Completion: {completion}")
    
    print("\n=== Question Answering ===")
    question = "What is React?"
    answer = engine.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    print("\n=== Code Explanation ===")
    code = "const [state, setState] = useState(0);"
    explanation = engine.explain_code(code)
    print(f"Code: {code}")
    print(f"Explanation: {explanation}")
