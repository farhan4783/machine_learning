"""
Training Script for Web Development LLM
Implements the complete training loop with checkpointing and logging
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from pathlib import Path
import json
from tqdm import tqdm
import time
from typing import Optional

from model import WebDevLLM
from tokenizer import WebDevTokenizer
from dataset import WebDevDataModule, collate_fn
from config import ModelConfig, TrainingConfig, DataConfig, SystemConfig


class Trainer:
    """Trainer class for Web Development LLM"""
    
    def __init__(
        self,
        model: WebDevLLM,
        tokenizer: WebDevTokenizer,
        data_module: WebDevDataModule,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_token_ids['pad_token'])
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=DataConfig.tensorboard_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.lr_scheduler == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_epochs
            )
        else:
            return None
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    
                    # Reshape for loss calculation
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                self.writer.add_scalar('train/loss', loss.item() * self.config.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()
                
                # Validation
                if self.global_step % self.config.eval_every_n_steps == 0:
                    val_loss = self.validate()
                    self.model.train()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/perplexity', perplexity, self.global_step)
        
        print(f"\nValidation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(is_best=True)
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        train_loader = self.data_module.train_dataloader()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Generate sample
            self.generate_sample()
        
        print("\nTraining complete!")
        self.writer.close()
    
    @torch.no_grad()
    def generate_sample(self):
        """Generate a sample text to monitor training progress"""
        self.model.eval()
        
        prompts = [
            "function",
            "const App =",
            ".container {",
            "import React"
        ]
        
        prompt = prompts[self.current_epoch % len(prompts)]
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate
        generated = self.model.generate(
            input_ids,
            max_length=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.special_token_ids['eos_token']
        )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].tolist())
        
        print(f"\n=== Generated Sample ===")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("=" * 50)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': {
                'model': vars(ModelConfig),
                'training': vars(TrainingConfig),
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = DataConfig.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = DataConfig.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N"""
        checkpoints = sorted(
            DataConfig.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                checkpoint.unlink()
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, model, tokenizer, data_module, config, device):
        """Load checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        trainer = cls(model, tokenizer, data_module, config, device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.current_epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {trainer.current_epoch}, step {trainer.global_step}")
        
        return trainer


def main():
    """Main training function"""
    # Set random seed
    torch.manual_seed(SystemConfig.seed)
    
    # Setup directories
    from config import setup_directories
    setup_directories()
    
    print("Initializing training...")
    
    # Load or create tokenizer
    tokenizer_path = DataConfig.tokenizer_dir
    if (tokenizer_path / "vocab.json").exists():
        print("Loading existing tokenizer...")
        tokenizer = WebDevTokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        # Load training data for tokenizer
        import json
        with open(DataConfig.processed_data_dir / "train.json", 'r') as f:
            train_data = json.load(f)
        
        texts = [item['text'] for item in train_data]
        tokenizer = WebDevTokenizer(vocab_size=ModelConfig.vocab_size)
        tokenizer.train(texts, verbose=True)
        tokenizer.save(tokenizer_path)
    
    # Update model config with actual vocab size
    ModelConfig.vocab_size = len(tokenizer)
    
    # Create model
    print("\nCreating model...")
    model = WebDevLLM(
        vocab_size=ModelConfig.vocab_size,
        d_model=ModelConfig.d_model,
        n_layers=ModelConfig.n_layers,
        n_heads=ModelConfig.n_heads,
        d_ff=ModelConfig.d_ff,
        max_seq_length=ModelConfig.max_seq_length,
        dropout=ModelConfig.dropout,
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create data module
    print("\nSetting up data...")
    data_module = WebDevDataModule(
        train_path=DataConfig.processed_data_dir / "train.json",
        val_path=DataConfig.processed_data_dir / "val.json",
        test_path=DataConfig.processed_data_dir / "test.json",
        tokenizer=tokenizer,
        batch_size=TrainingConfig.batch_size,
        max_length=ModelConfig.max_seq_length,
        num_workers=SystemConfig.num_workers,
    )
    data_module.setup()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_module=data_module,
        config=TrainingConfig,
        device=SystemConfig.device
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
