# Web Development LLM Model

A transformer-based language model trained from scratch on comprehensive web development knowledge. This model generates educational knowledge cards covering all aspects of web development.

## Project Structure

```
webdev-llm/
├── src/                    # Core ML source code
│   ├── model.py           # Transformer architecture
│   ├── tokenizer.py       # Custom tokenizer
│   ├── dataset.py         # Data loading and preprocessing
│   ├── train.py           # Training loop
│   ├── config.py          # Configuration
│   ├── card_generator.py  # Card generation system
│   └── inference.py       # Model inference utilities
├── data/
│   ├── raw/               # Raw collected data
│   └── processed/         # Preprocessed training data
├── models/
│   └── checkpoints/       # Model checkpoints
├── api/                   # FastAPI backend
│   ├── main.py
│   └── routes.py
├── tests/                 # Unit tests
├── logs/                  # Training logs
└── requirements.txt
```

## Features

- **Custom Transformer Model**: Built from scratch with multi-head attention
- **Web Dev Specialized**: Trained on HTML, CSS, JavaScript, frameworks, databases, DevOps
- **Knowledge Card Generation**: Generates structured educational cards on any web dev topic
- **Topic Categories**: Frontend, Backend, Database, DevOps, Tools
- **FastAPI Backend**: RESTful API for model inference

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended (RTX 3080/4080 or better)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for data and model checkpoints
- **CPU Training**: Possible but significantly slower

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```bash
python data/data_collector.py
```

### 2. Data Preprocessing
```bash
python data/preprocessor.py
```

### 3. Train Model
```bash
python src/train.py --epochs 10 --batch-size 32
```

### 4. Generate Cards
```bash
python src/card_generator.py --topic "React Hooks"
```

### 5. Run API Server
```bash
cd api
uvicorn main:app --reload
```

## Model Architecture

- **Type**: Transformer (GPT-style)
- **Parameters**: ~124M (configurable)
- **Layers**: 12
- **Attention Heads**: 12
- **Embedding Dimension**: 768
- **Context Length**: 1024 tokens

## Training Data Sources

- MDN Web Docs
- W3C Specifications
- Framework Documentation (React, Vue, Angular, Next.js)
- Backend Frameworks (Express, Django, Flask, FastAPI)
- Database Documentation (PostgreSQL, MongoDB, MySQL)
- DevOps Guides (Docker, Kubernetes, CI/CD)

## API Endpoints

- `POST /generate-card` - Generate a knowledge card
- `POST /generate-batch` - Generate multiple cards
- `GET /topics` - List available topics
- `GET /model-info` - Model metadata

## License

MIT
