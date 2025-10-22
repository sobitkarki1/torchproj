# PyTorch LSTM Implementation Guide

## 🎯 PyTorch-Specific Implementation Details

This guide provides PyTorch-specific details for implementing the LSTM model for NEPSE stock prediction.

---

## 🧠 Model Architecture

### Basic LSTM Model

```python
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """
    Multi-layer LSTM for stock price prediction
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.2, num_layers=2):
        super(StockLSTM, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
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
        self.fc2 = nn.Linear(hidden_sizes[2], 1)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        
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
        out = self.fc2(out)
        
        return out
```

### Model with Attention (Advanced)

```python
class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
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
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)
        
        # Output
        out = self.fc(context)
        return out
```

---

## 📊 PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    """
    PyTorch Dataset for stock price sequences
    """
    def __init__(self, data, sequence_length=30, prediction_horizon=5):
        """
        Args:
            data: DataFrame with stock data (features already calculated)
            sequence_length: Number of days to look back
            prediction_horizon: Days ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Create sequences
        self.X, self.y = self._create_sequences(data)
        
    def _create_sequences(self, data):
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.sequence_length].values)
            
            # Target (price after prediction_horizon days)
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            y.append(data.iloc[target_idx]['Close'])
        
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]])
        )
```

---

## 🎓 Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
    """
    Training loop with validation
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'models/best_model.pth')
            print(f'✅ Model saved with val_loss = {val_loss:.6f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses
```

---

## 🔮 Inference/Prediction

```python
def predict(model, data, scaler_X, scaler_y, sequence_length=30, device='cuda'):
    """
    Make predictions on new data
    """
    model.eval()
    model = model.to(device)
    
    # Prepare input
    if len(data) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} days of data")
    
    # Get last sequence
    last_sequence = data[-sequence_length:].values
    
    # Scale
    last_sequence_scaled = scaler_X.transform(last_sequence)
    
    # Convert to tensor
    X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prediction_scaled = model(X).cpu().numpy()
    
    # Inverse transform
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction[0][0]
```

---

## 📈 Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, test_loader, scaler_y, device='cuda'):
    """
    Evaluate model on test set
    """
    model.eval()
    model = model.to(device)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch).cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"📊 Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'actuals': actuals
    }
```

---

## 💾 Model Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.6f}")
    return model, optimizer, epoch
```

---

## ⚡ GPU Optimization Tips

### Check CUDA Availability

```python
import torch

def setup_device():
    """
    Setup device and print info
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU not available, using CPU")
    
    return device
```

### Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Enable mixed precision training (faster)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Optimal Batch Size

```python
# For GTX 1650 Ti (4GB VRAM)
BATCH_SIZE = 256  # Adjust based on sequence length and model size

# If out of memory, reduce batch size:
BATCH_SIZE = 128
# or
BATCH_SIZE = 64
```

---

## 🔬 Feature Engineering

```python
def add_technical_indicators(df):
    """
    Add technical indicators to DataFrame
    """
    df = df.copy()
    
    # Returns
    df['return'] = df['Close'].pct_change()
    df['volume_change'] = df['Volume'].pct_change()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential moving averages
    df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # Volatility
    df['std_5'] = df['Close'].rolling(window=5).std()
    df['std_10'] = df['Close'].rolling(window=10).std()
    df['std_20'] = df['Close'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    
    # Drop NaN values
    df = df.dropna()
    
    return df
```

---

## 📊 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(train_losses, val_losses):
    """
    Plot training and validation loss
    """
    plt.figure(figsize=(12, 5))
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(actuals, predictions, stock_name='Stock'):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(14, 6))
    
    plt.plot(actuals, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price (NPR)')
    plt.title(f'{stock_name} - Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_distribution(actuals, predictions):
    """
    Plot error distribution
    """
    errors = actuals - predictions
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error (NPR)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], 
             [actuals.min(), actuals.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Price (NPR)')
    plt.ylabel('Predicted Price (NPR)')
    plt.title('Actual vs Predicted Scatter')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

---

## 🚀 Complete Example Pipeline

```python
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Setup device
device = setup_device()

# 2. Load and prepare data
df = pd.read_csv('data/raw/NABIL_2000-01-01_2021-12-31.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 3. Add technical indicators
df = add_technical_indicators(df)

# 4. Select features
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'return', 'volume_change', 'sma_5', 'sma_10', 'sma_20',
                'std_5', 'std_10', 'rsi', 'macd']
features = df[feature_cols]

# 5. Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(df[['Close']])

# 6. Create dataset
dataset = StockDataset(pd.DataFrame(X_scaled, columns=feature_cols), 
                       sequence_length=30, prediction_horizon=5)

# 7. Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 8. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 9. Initialize model
model = StockLSTM(input_size=len(feature_cols), 
                  hidden_sizes=[128, 64, 32], 
                  dropout=0.2)

# 10. Train model
train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                       num_epochs=100, lr=0.001, device=device)

# 11. Evaluate
results = evaluate_model(model, val_loader, scaler_y, device=device)

# 12. Plot results
plot_training_history(train_losses, val_losses)
plot_predictions(results['actuals'], results['predictions'], 'NABIL')
plot_error_distribution(results['actuals'], results['predictions'])
```

---

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Time Series Forecasting with PyTorch](https://pytorch.org/tutorials/beginner/timeseries_tutorial.html)
- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

**Happy Training! 🚀**
