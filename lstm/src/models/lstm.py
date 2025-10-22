"""
LSTM Model Architecture for Stock Price Prediction
"""

import torch
import torch.nn as nn
from typing import List, Optional


class StockLSTM(nn.Module):
    """
    Multi-layer LSTM for stock price prediction
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout: float = 0.2,
                 num_layers: int = 2):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
            num_layers: Number of LSTM layers per stack
        """
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.num_layers = num_layers
        
        # LSTM Layer 1
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM Layer 2
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            num_layers=1,
            batch_first=True
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_sizes[2], 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, 1)
        """
        # LSTM layer 1
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        # LSTM layer 2
        lstm_out, (hidden, cell) = self.lstm2(lstm_out)
        
        # Use the last hidden state
        out = hidden[-1]  # Shape: (batch, hidden_size)
        out = self.dropout2(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for stock price prediction
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize Attention LSTM
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        Forward pass with attention
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, 1)
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)
        
        # Output
        out = self.fc(context)
        return out
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_type: str = 'basic',
                 input_size: int = 24,
                 **kwargs) -> nn.Module:
    """
    Factory function to create LSTM models
    
    Args:
        model_type: 'basic' or 'attention'
        input_size: Number of input features
        **kwargs: Additional model parameters
        
    Returns:
        LSTM model instance
    """
    if model_type == 'basic':
        model = StockLSTM(input_size=input_size, **kwargs)
    elif model_type == 'attention':
        model = AttentionLSTM(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    """Test model creation and forward pass"""
    print("=" * 70)
    print("TESTING LSTM MODELS")
    print("=" * 70)
    
    # Model parameters
    input_size = 24  # Number of features
    batch_size = 32
    sequence_length = 30
    
    # Test basic LSTM
    print("\n1. Testing Basic LSTM Model")
    print("-" * 70)
    model = create_model('basic', input_size=input_size)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"✓ Forward pass successful")
    
    # Test attention LSTM
    print("\n2. Testing Attention LSTM Model")
    print("-" * 70)
    model_attn = create_model('attention', input_size=input_size, hidden_size=128, num_layers=2)
    print(f"Model created: {model_attn.__class__.__name__}")
    print(f"Total parameters: {model_attn.count_parameters():,}")
    
    output_attn = model_attn(dummy_input)
    print(f"Output shape: {output_attn.shape}")
    print(f"✓ Forward pass successful")
    
    # GPU test
    if torch.cuda.is_available():
        print("\n3. Testing GPU Compatibility")
        print("-" * 70)
        device = torch.device('cuda')
        model_gpu = model.to(device)
        dummy_input_gpu = dummy_input.to(device)
        
        output_gpu = model_gpu(dummy_input_gpu)
        print(f"Model on device: {next(model_gpu.parameters()).device}")
        print(f"Input on device: {dummy_input_gpu.device}")
        print(f"Output on device: {output_gpu.device}")
        print(f"✓ GPU test successful")
        
        # Check memory
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("\nGPU not available, using CPU")
    
    # Model summary
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("\nBasic LSTM:")
    print(model)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Create training script: python src/training/trainer.py")
    print("2. Train model on stock data")
    print("=" * 70)


if __name__ == "__main__":
    main()
