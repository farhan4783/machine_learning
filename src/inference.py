"""
Inference utilities for the Web Development LLM
"""

import torch
from typing import List, Optional
from pathlib import Path

from model import WebDevLLM
from tokenizer import WebDevTokenizer
from config import ModelConfig, InferenceConfig, DataConfig
from ai_engine import WebDevAIEngine


class InferenceEngine:
    """Inference engine for text generation and web development assistance"""
    
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
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from a prompt
        """
        # If prompt contains code or questions, use intelligent domain answering
        if "function" in prompt or "const " in prompt or "class " in prompt or "<" in prompt:
            return [self.ai_engine.complete_code(prompt)]
        elif "explain" in prompt.lower():
            return [self.ai_engine.explain_code(prompt)]
        elif "user:" in prompt.lower() or "?" in prompt:
            q = prompt.replace("User:", "").replace("Assistant:", "").strip()
            return [self.ai_engine.answer_question(q)]
            
        # Fallback to standard intelligent response
        return [self.ai_engine.answer_question(prompt)]
    
    def complete_code(self, code_snippet: str, max_length: int = 256) -> str:
        """Complete a code snippet accurately"""
        return self.ai_engine.complete_code(code_snippet)
    
    def explain_code(self, code: str) -> str:
        """Explain what a code snippet does in detail"""
        return self.ai_engine.explain_code(code)
    
    def answer_question(self, question: str) -> str:
        """Answer a web development question like a senior engineer"""
        return self.ai_engine.answer_question(question)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: Optional[str] = None) -> 'InferenceEngine':
        """Create inference engine from checkpoint"""
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
