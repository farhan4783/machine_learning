"""
Configuration file for the Web Development LLM Model
Contains all hyperparameters and settings for training and inference
"""

import torch
from pathlib import Path

class ModelConfig:
    """Model architecture configuration"""
    
    # Model Architecture
    vocab_size = 50000  # Will be updated after tokenizer training
    d_model = 768  # Embedding dimension
    n_layers = 12  # Number of transformer layers
    n_heads = 12  # Number of attention heads
    d_ff = 3072  # Feed-forward dimension (4 * d_model)
    max_seq_length = 1024  # Maximum sequence length
    dropout = 0.1
    
    # Special tokens
    pad_token = "<PAD>"
    unk_token = "<UNK>"
    bos_token = "<BOS>"  # Beginning of sequence
    eos_token = "<EOS>"  # End of sequence
    
    # Web development specific tokens
    code_start_token = "<CODE>"
    code_end_token = "</CODE>"
    html_token = "<HTML>"
    css_token = "<CSS>"
    js_token = "<JS>"


class TrainingConfig:
    """Training configuration"""
    
    # Training hyperparameters
    batch_size = 32
    learning_rate = 5e-4
    weight_decay = 0.01
    num_epochs = 10
    warmup_steps = 1000
    max_grad_norm = 1.0
    
    # Optimization
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    
    # Learning rate schedule
    lr_scheduler = "cosine"  # Options: "cosine", "linear", "constant"
    
    # Gradient accumulation
    gradient_accumulation_steps = 4
    
    # Mixed precision training
    use_amp = True  # Automatic Mixed Precision
    
    # Checkpointing
    save_every_n_steps = 1000
    eval_every_n_steps = 500
    keep_last_n_checkpoints = 3
    
    # Early stopping
    patience = 3
    min_delta = 0.001


class DataConfig:
    """Data configuration"""
    
    # Data paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    
    # Model paths
    models_dir = base_dir / "models"
    checkpoint_dir = models_dir / "checkpoints"
    tokenizer_dir = models_dir / "tokenizer"
    
    # Logs
    logs_dir = base_dir / "logs"
    tensorboard_dir = logs_dir / "tensorboard"
    
    # Data splits
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    
    # Data processing
    min_text_length = 50  # Minimum characters per sample
    max_text_length = 10000  # Maximum characters per sample
    
    # Web development topics
    topics = [
        # Frontend
        "HTML", "CSS", "JavaScript", "TypeScript",
        "React", "Vue", "Angular", "Svelte",
        "Next.js", "Nuxt.js", "Gatsby",
        "Tailwind CSS", "Bootstrap", "Material-UI",
        "Webpack", "Vite", "Rollup",
        
        # Backend
        "Node.js", "Express", "Nest.js",
        "Python", "Django", "Flask", "FastAPI",
        "PHP", "Laravel",
        "Ruby", "Rails",
        "Java", "Spring Boot",
        
        # Databases
        "SQL", "PostgreSQL", "MySQL", "SQLite",
        "MongoDB", "Redis", "Elasticsearch",
        "Prisma", "TypeORM", "Sequelize",
        
        # DevOps & Tools
        "Git", "GitHub", "GitLab",
        "Docker", "Kubernetes",
        "CI/CD", "Jenkins", "GitHub Actions",
        "AWS", "Azure", "Google Cloud",
        "Nginx", "Apache",
        
        # Testing
        "Jest", "Mocha", "Pytest",
        "Selenium", "Cypress", "Playwright",
        
        # APIs
        "REST API", "GraphQL", "WebSockets",
        "OAuth", "JWT", "Authentication",
    ]


class InferenceConfig:
    """Inference and card generation configuration"""
    
    # Generation parameters
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    max_length = 512
    num_return_sequences = 1
    
    # Card structure
    card_sections = [
        "title",
        "description",
        "key_concepts",
        "code_example",
        "use_cases",
        "best_practices",
        "common_pitfalls",
        "resources"
    ]


class SystemConfig:
    """System configuration"""
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # For data loading
    
    # Random seed for reproducibility
    seed = 42
    
    # Logging
    log_level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Create directories if they don't exist
def setup_directories():
    """Create necessary directories"""
    dirs = [
        DataConfig.raw_data_dir,
        DataConfig.processed_data_dir,
        DataConfig.checkpoint_dir,
        DataConfig.tokenizer_dir,
        DataConfig.tensorboard_dir,
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_directories()
    print(f"Device: {SystemConfig.device}")
    print(f"Model parameters: ~{ModelConfig.n_layers * ModelConfig.d_model * ModelConfig.d_ff // 1_000_000}M")
