# Web Development LLM - Quick Start Guide

This guide will help you get started with training and using the Web Development LLM.

## Prerequisites

- Python 3.8+
- NVIDIA GPU with 16GB+ VRAM (recommended) or CPU
- 32GB+ RAM
- 50GB+ free disk space

## Installation

1. **Create virtual environment**
```bash
python -m venv venv
```

2. **Activate virtual environment**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Step-by-Step Workflow

### Step 1: Collect Training Data

```bash
python data/data_collector.py
```

This will:
- Collect web development documentation (currently uses synthetic examples for testing)
- Save raw data to `data/raw/raw_data.json`

**Note**: For production, you can uncomment the MDN scraping code to collect real documentation.

### Step 2: Preprocess Data

```bash
python data/preprocessor.py
```

This will:
- Clean and normalize text
- Process code examples
- Augment data with Q&A formats
- Split into train/val/test sets
- Save to `data/processed/`

### Step 3: Train the Model

```bash
python src/train.py
```

This will:
- Train or load tokenizer
- Create the transformer model (~124M parameters)
- Train for 10 epochs (configurable in `src/config.py`)
- Save checkpoints to `models/checkpoints/`
- Log training progress to TensorBoard

**Monitor training with TensorBoard**:
```bash
tensorboard --logdir logs/tensorboard
```

### Step 4: Generate Knowledge Cards

After training, generate cards:

```bash
python src/card_generator.py
```

This will generate sample cards for various web development topics.

### Step 5: Run the API Server

```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage Examples

### Generate a Single Card

```bash
curl -X POST "http://localhost:8000/generate-card" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "React Hooks",
    "card_type": "concept",
    "max_length": 512,
    "temperature": 0.8
  }'
```

### Generate Multiple Cards

```bash
curl -X POST "http://localhost:8000/generate-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "topics": ["React Hooks", "CSS Flexbox", "Node.js"],
    "card_type": "code_example"
  }'
```

### Generate Custom Text

```bash
curl -X POST "http://localhost:8000/generate-text" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain how to use async/await in JavaScript",
    "max_length": 512,
    "temperature": 0.8
  }'
```

### Get Available Topics

```bash
curl "http://localhost:8000/topics"
```

### Get Model Information

```bash
curl "http://localhost:8000/model-info"
```

## Configuration

Edit `src/config.py` to customize:

- **Model Architecture**: layers, heads, dimensions
- **Training**: batch size, learning rate, epochs
- **Data**: topics, splits, paths
- **Inference**: temperature, sampling parameters

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Reduce `batch_size` in `TrainingConfig`
2. Reduce `max_seq_length` in `ModelConfig`
3. Enable gradient accumulation (already configured)
4. Use CPU training (slower but no memory limit)

### Model Not Found

If the API can't find the model:
1. Ensure you've trained the model first
2. Check that `models/checkpoints/best_model.pt` exists
3. Verify the path in `config.py`

### Slow Training

Training speed depends on:
- GPU: RTX 3080/4080 recommended
- Batch size: Larger = faster but more memory
- Mixed precision: Enabled by default for speed

## Next Steps

1. **Collect More Data**: Uncomment MDN scraping or add your own data sources
2. **Train Longer**: Increase `num_epochs` for better results
3. **Fine-tune**: Adjust hyperparameters in `config.py`
4. **Build Frontend**: Create a UI to interact with the API
5. **Deploy**: Deploy the API to a cloud service

## File Structure

```
webdev-llm/
├── src/
│   ├── model.py           # Transformer architecture
│   ├── tokenizer.py       # BPE tokenizer
│   ├── dataset.py         # Data loading
│   ├── train.py           # Training script
│   ├── config.py          # Configuration
│   ├── card_generator.py  # Card generation
│   └── inference.py       # Inference utilities
├── data/
│   ├── data_collector.py  # Data collection
│   ├── preprocessor.py    # Data preprocessing
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── api/
│   ├── main.py           # FastAPI application
│   └── routes.py         # API routes
├── models/
│   ├── checkpoints/      # Model checkpoints
│   └── tokenizer/        # Tokenizer files
└── logs/                 # Training logs
```

## Support

For issues or questions, refer to:
- `README.md` for project overview
- `src/config.py` for configuration options
- API docs at `http://localhost:8000/docs` (when running)


I've completed the requested fixes and verified them! Here is a summary of the accomplishments:

Fixed the Autoregressive ML Bug: Adjusted label shifting in src/train.py. The model will now correctly learn to predict the next token instead of trying to map identity inputs.
Added Model Intelligence: Updated data/preprocessor.py and src/inference.py with standard User / Assistant conversational wrappers so the model learns structured Q&A interactions.
Fixed Flutter Compilation: Removed outdated pulse() animation hooks and missing LucideIcons across the UI files, replacing them with standard Material icons. The Flutter frontend now compiles and runs flawlessly without errors.
Added Guide: Wrote starting_guide.md to the root folder instructing you exactly how to train, run, and host the ML backend and test the Flutter UI!