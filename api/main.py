"""
FastAPI Backend for Web Development LLM
Provides REST API for model inference and card generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import torch

from card_generator import CardGenerator
from inference import InferenceEngine
from config import DataConfig, InferenceConfig


# Pydantic models for request/response
class GenerateCardRequest(BaseModel):
    topic: str = Field(..., description="Web development topic")
    card_type: str = Field("concept", description="Type of card to generate")
    max_length: int = Field(512, description="Maximum generation length")
    temperature: float = Field(0.8, description="Sampling temperature")


class GenerateCardResponse(BaseModel):
    topic: str
    type: str
    title: str
    content: str
    code_examples: Optional[List[str]] = None


class GenerateBatchRequest(BaseModel):
    topics: List[str] = Field(..., description="List of topics")
    card_type: str = Field("concept", description="Type of cards to generate")


class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    max_length: int = Field(512, description="Maximum generation length")
    temperature: float = Field(0.8, description="Sampling temperature")
    top_k: int = Field(50, description="Top-k sampling")
    top_p: float = Field(0.95, description="Nucleus sampling")


class GenerateTextResponse(BaseModel):
    prompt: str
    generated_text: str


class TopicInfo(BaseModel):
    name: str
    category: str
    description: str


class ModelInfo(BaseModel):
    model_name: str
    parameters: int
    vocab_size: int
    max_sequence_length: int
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="Web Development LLM API",
    description="API for generating web development knowledge cards and text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model instances
card_generator: Optional[CardGenerator] = None
inference_engine: Optional[InferenceEngine] = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global card_generator, inference_engine
    
    checkpoint_path = DataConfig.checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        print("WARNING: No trained model found. Please train the model first.")
        return
    
    print("Loading models...")
    card_generator = CardGenerator.load_model(checkpoint_path, device=device)
    inference_engine = InferenceEngine.from_checkpoint(checkpoint_path, device=device)
    print(f"Models loaded successfully on {device}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Web Development LLM API",
        "version": "1.0.0",
        "endpoints": {
            "generate_card": "/generate-card",
            "generate_batch": "/generate-batch",
            "generate_text": "/generate-text",
            "topics": "/topics",
            "model_info": "/model-info"
        }
    }


@app.post("/generate-card", response_model=GenerateCardResponse)
async def generate_card(request: GenerateCardRequest):
    """Generate a knowledge card for a topic"""
    if card_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        card = card_generator.generate_card(
            topic=request.topic,
            card_type=request.card_type,
            max_length=request.max_length,
            temperature=request.temperature,
        )
        
        return GenerateCardResponse(**card)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-batch")
async def generate_batch(request: GenerateBatchRequest):
    """Generate multiple knowledge cards"""
    if card_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        cards = card_generator.generate_batch(
            topics=request.topics,
            card_type=request.card_type,
        )
        
        return {"cards": cards}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-text", response_model=GenerateTextResponse)
async def generate_text(request: GenerateTextRequest):
    """Generate text from a prompt"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_texts = inference_engine.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        
        return GenerateTextResponse(
            prompt=request.prompt,
            generated_text=generated_texts[0]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics")
async def get_topics():
    """Get list of available web development topics"""
    topics = {
        "Frontend": [
            {"name": "HTML", "category": "Frontend", "description": "HyperText Markup Language"},
            {"name": "CSS", "category": "Frontend", "description": "Cascading Style Sheets"},
            {"name": "JavaScript", "category": "Frontend", "description": "Programming language for web"},
            {"name": "React", "category": "Frontend", "description": "JavaScript library for UI"},
            {"name": "Vue", "category": "Frontend", "description": "Progressive JavaScript framework"},
            {"name": "Angular", "category": "Frontend", "description": "TypeScript-based framework"},
        ],
        "Backend": [
            {"name": "Node.js", "category": "Backend", "description": "JavaScript runtime"},
            {"name": "Express", "category": "Backend", "description": "Node.js web framework"},
            {"name": "Django", "category": "Backend", "description": "Python web framework"},
            {"name": "Flask", "category": "Backend", "description": "Python microframework"},
            {"name": "FastAPI", "category": "Backend", "description": "Modern Python API framework"},
        ],
        "Database": [
            {"name": "SQL", "category": "Database", "description": "Structured Query Language"},
            {"name": "PostgreSQL", "category": "Database", "description": "Advanced relational database"},
            {"name": "MongoDB", "category": "Database", "description": "NoSQL document database"},
            {"name": "Redis", "category": "Database", "description": "In-memory data store"},
        ],
        "DevOps": [
            {"name": "Git", "category": "DevOps", "description": "Version control system"},
            {"name": "Docker", "category": "DevOps", "description": "Containerization platform"},
            {"name": "Kubernetes", "category": "DevOps", "description": "Container orchestration"},
            {"name": "CI/CD", "category": "DevOps", "description": "Continuous Integration/Deployment"},
        ]
    }
    
    return topics


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if card_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="WebDevLLM",
        parameters=card_generator.model.count_parameters(),
        vocab_size=len(card_generator.tokenizer),
        max_sequence_length=card_generator.model.max_seq_length,
        device=device
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": card_generator is not None,
        "device": device
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
