# Web Development LLM Model & AI Studio

A domain-specialized transformer language model trained from scratch on comprehensive full-stack web development knowledge. The project features an end-to-end Machine Learning pipeline, a high-performance FastAPI backend, and a modern React Web Studio for educational knowledge card synthesis and code assistance.

## Project Structure

```
webdev-llm/
├── src/                    # Core ML source code
│   ├── model.py           # Transformer architecture (RMSNorm, RoPE, SwiGLU, KV Cache)
│   ├── tokenizer.py       # Custom BPE tokenizer from scratch
│   ├── dataset.py         # PyTorch Dataset & DataLoader pipelines
│   ├── train.py           # Training loop with AMP & gradient accumulation
│   ├── config.py          # Architecture & training hyperparameters
│   ├── card_generator.py  # Structured knowledge card generator
│   └── inference.py       # Autoregressive generation & code explanation
├── data/
│   ├── data_collector.py  # Scrapes MDN & builds synthetic webdev corpus
│   ├── preprocessor.py    # Text cleaning, code normalization & dataset splits
│   ├── raw/               # Raw collected data
│   └── processed/         # train.json, val.json, test.json
├── models/
│   ├── checkpoints/       # Trained model weights (best_model.pt)
│   └── tokenizer/         # BPE vocabulary & merges
├── api/                   # FastAPI backend
│   ├── main.py            # REST API & React frontend server
│   └── routes.py          # API route definitions
├── frontend/              # Modern React Web Studio
│   ├── index.html         # HTML5 shell with Google Fonts & Lucide icons
│   ├── app.jsx            # Interactive React application
│   ├── style.css          # Glassmorphism dark mode design system
│   └── package.json       # React frontend metadata
├── tests/                 # Pytest test suite
└── requirements.txt
```

## Features

- **Custom Transformer Architecture**: RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU activation, and Key-Value (KV) cache for $O(1)$ recurrent step latency.
- **BPE Tokenizer**: Built from scratch with web dev specific tokens (`<CODE>`, `<HTML>`, `<CSS>`, `<JS>`).
- **Educational Knowledge Card Generator**: Produces structured cards covering concepts, code snippets, best practices, and use cases.
- **Interactive React Web Studio**: Beautiful dark-mode UI with live card generation, categorized topic directory, code playground, and system diagnostics.
- **Unified FastAPI Backend**: High-throughput REST API serving both ML endpoints and the React single-page application.

## Quickstart

```bash
# 1. Activate virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Collect and preprocess training data
python data/data_collector.py
python data/preprocessor.py

# 3. Train the model & tokenizer
python test_train.py

# 4. Start the Web Studio & API Server
python -m uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 --reload
```

Navigate to **`http://localhost:8000`** to access the React Web Studio!

## API Endpoints

- `GET /` - React Web Studio GUI
- `POST /generate-card` - Generate a structured knowledge card
- `POST /generate-batch` - Generate multiple cards in batch
- `POST /generate-text` - Free-form prompt completion and Q&A
- `GET /topics` - Categorized web development topic catalog
- `GET /model-info` - Active model parameters and architecture metadata
- `GET /health` - Backend and compute device health check

## License

MIT

