"""
Training Script for LSTM Stock Price Prediction
Complete training pipeline with validation and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lstm import create_model
from src.data.dataset import StockDataset, create_dataloaders
import pandas as pd


class LSTMTrainer:
    """Trainer class for LSTM models"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer
        
        Args:
            model: LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in tqdm(self.train_loader, desc='Training'):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.val_loader, desc='Validating'):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self,
              num_epochs: int = 100,
              patience: int = 15,
              save_dir: str = 'models/checkpoints'):
        """
        Complete training loop
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        patience_counter = 0
        
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                print(f"✓ Best model saved (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final train loss: {self.train_losses[-1]:.6f}")
        print(f"Final val loss: {self.val_losses[-1]:.6f}")
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss': loss,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f}")


def setup_device():
    """Setup compute device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    return device


def main():
    """Main training function"""
    print("=" * 70)
    print("LSTM STOCK PRICE PREDICTION - TRAINING")
    print("=" * 70)
    
    # Configuration
    import pathlib
    script_dir = pathlib.Path(__file__).parent.parent.parent
    PROCESSED_DATA = str(script_dir / "data" / "processed" / "NABIL_processed.csv")
    SEQUENCE_LENGTH = 30
    PREDICTION_HORIZON = 5
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Setup device
    device = setup_device()
    
    # Load processed data
    print(f"\nLoading processed data: {PROCESSED_DATA}")
    if not os.path.exists(PROCESSED_DATA):
        print(f"Error: {PROCESSED_DATA} not found!")
        print("Please run: python src/data/preprocessor.py first")
        return
    
    df = pd.read_csv(PROCESSED_DATA)
    print(f"Data loaded: {df.shape}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = StockDataset(
        data=df,
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        target_column='Close Price'
    )
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Features: {dataset.get_feature_size()}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        dataset,
        train_split=0.8,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type='basic',
        input_size=dataset.get_feature_size(),
        hidden_sizes=[128, 64, 32],
        dropout=0.2,
        num_layers=2
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = LSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    train_losses, val_losses = trainer.train(
        num_epochs=NUM_EPOCHS,
        patience=15,
        save_dir='models/checkpoints'
    )
    
    # Save final model
    print("\nSaving final model...")
    torch.save(model.state_dict(), 'models/lstm_final.pth')
    print("✓ Model saved to: models/lstm_final.pth")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Evaluate model performance")
    print("2. Make predictions on test data")
    print("3. Visualize results")
    print("=" * 70)


if __name__ == "__main__":
    main()
