import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import TrainingConfig
TrainingConfig.num_epochs = 1
TrainingConfig.batch_size = 2
TrainingConfig.gradient_accumulation_steps = 1
TrainingConfig.save_every_n_steps = 100
TrainingConfig.eval_every_n_steps = 100

from train import main

if __name__ == "__main__":
    main()
