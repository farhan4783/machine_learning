import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelConfig, TrainingConfig
ModelConfig.d_model = 256
ModelConfig.n_layers = 4
ModelConfig.n_heads = 4
ModelConfig.d_ff = 1024
ModelConfig.max_seq_length = 256
ModelConfig.vocab_size = 5000

TrainingConfig.num_epochs = 1
TrainingConfig.batch_size = 16
TrainingConfig.gradient_accumulation_steps = 1
TrainingConfig.save_every_n_steps = 50
TrainingConfig.eval_every_n_steps = 50

from train import main

if __name__ == "__main__":
    main()

