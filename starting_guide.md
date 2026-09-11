# WebDev LLM Project - Starting Guide

Welcome! This starting guide explains how to spin up both the Machine Learning backend (WebDev LLM Transformer) and the modern React Web Frontend.

## 1. Prerequisites
- **Python**: 3.8 to 3.14.
- **Web Browser**: Chrome, Edge, Firefox, or Safari (no Node.js installation required to run the Web Studio).
- **Compute Device**: Automatically supports both CPU and NVIDIA CUDA GPUs.

---

## 2. Server/Backend Setup (Python)

The backend powers the WebDev LLM model training, inference, and the unified FastAPI application that serves both the REST API and the React Web Studio.

### A. Environment Setup
1. Open a terminal in the project root directory:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### B. Collecting & Preprocessing Data
1. Collect synthetic and web development documentation:
   ```bash
   python data/data_collector.py
   ```
2. Run data cleaning, normalization, Q&A augmentation, and split generation:
   ```bash
   python data/preprocessor.py
   ```
   This creates `train.json`, `val.json`, and `test.json` under `data/processed/`.

### C. Training the Model
To train the model:
- **Fast Local / Dev Training**:
  ```bash
  python test_train.py
  ```
- **Full Production Training (124M Parameters)**:
  ```bash
  python src/train.py
  ```
This produces the trained checkpoint `models/checkpoints/best_model.pt` and custom BPE tokenizer `models/tokenizer/`.

### D. Running the React Web Studio & API Server
Start the unified FastAPI server:
```bash
python -m uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 --reload
```
or
```bash
cd api
python main.py
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 3. React Web Frontend Features

The React Web Frontend (`frontend/`) is served directly by the backend at `http://localhost:8000` and can also be opened standalone in `frontend/index.html`:

- **🎴 Knowledge Card Generator**: Synthesize educational cards on any web dev subject (React, CSS Flexbox, Node.js, SQL, Docker, TypeScript) with customizable card types (Concept, Code Example, Tutorial, Best Practices, Tradeoffs).
- **📚 Interactive Topic Explorer**: Browse categorized full-stack topics with 1-click card generation.
- **💻 AI Playground**: Test raw code completion, code explanation, and conversational Q&A.
- **⚙️ Model Diagnostics**: Inspect active device (CPU/CUDA), parameter counts, vocabulary dimensions, and neural architecture (RoPE, RMSNorm, SwiGLU, KV Cache).
- **📥 Export Options**: Copy to clipboard, download as Markdown (`.md`), or download as JSON.

---

## 4. Running Unit Tests
Verify model architecture and tokenizer correctness:
```bash
pytest tests/
```

