# 🚀 Quick Reference Guide

**LSTM Project - Common Commands and Workflows**

---

## 🖥️ Environment Commands

### Check PyTorch and CUDA:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Expected Output:
```
PyTorch: 2.6.0+cu126
CUDA: True
GPU: NVIDIA GeForce GTX 1650 Ti
```

---

## 📁 Project Navigation

### Key Directories:
```bash
cd c:\Users\Asus\Development\torchproj\lstm

# Data
cd data/raw         # Raw stock CSV files (400+ stocks)
cd data/processed   # Preprocessed data (to be created)

# Source code
cd src              # Python modules

# Documentation
# README.md           - Project overview
# PYTORCH_GUIDE.md    - Complete implementation guide
# PROJECT_STATUS.md   - Current status and roadmap
# QUICK_REFERENCE.md  - This file
```

---

## 📊 Data Exploration

### Load a Sample Stock:
```python
import pandas as pd

# Load NABIL stock data
df = pd.read_csv('data/raw/NABIL_2000-01-01_2021-12-31.csv')
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nColumns: {df.columns.tolist()}")
```

### List All Stocks:
```python
import os

stock_files = sorted([f for f in os.listdir('data/raw') if f.endswith('.csv')])
print(f"Total stocks: {len(stock_files)}")
print("\nFirst 10 stocks:")
for i, f in enumerate(stock_files[:10], 1):
    print(f"{i}. {f}")
```

---

## 🧠 Model Development

### Create Dataset:
```python
from torch.utils.data import Dataset, DataLoader
import torch

class StockDataset(Dataset):
    def __init__(self, data, sequence_length=30):
        self.data = data
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length]
        return torch.FloatTensor(X), torch.FloatTensor([y])
```

### Basic LSTM Model:
```python
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (h, c) = self.lstm(x)
        out = self.fc(h[-1])
        return out
```

### Train Model:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLSTM(input_size=14).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 💾 Save and Load Models

### Save Model:
```python
# Save complete model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'models/lstm_checkpoint.pth')
```

### Load Model:
```python
checkpoint = torch.load('models/lstm_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## 📈 Common Data Operations

### Add Technical Indicators:
```python
# Moving Average
df['sma_5'] = df['Close'].rolling(window=5).mean()
df['sma_10'] = df['Close'].rolling(window=10).mean()
df['sma_20'] = df['Close'].rolling(window=20).mean()

# Volatility
df['std_10'] = df['Close'].rolling(window=10).std()

# Returns
df['return'] = df['Close'].pct_change()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))
```

### Normalize Data:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Close', 'Volume', 'return']])
```

---

## 🎯 Evaluation

### Calculate Metrics:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predictions
predictions = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
```

---

## 🔧 Troubleshooting

### GPU Out of Memory:
```python
# Reduce batch size
BATCH_SIZE = 64  # Instead of 256

# Clear cache
torch.cuda.empty_cache()

# Enable gradient checkpointing
# (saves memory at cost of speed)
```

### Data Loading Issues:
```python
# Check file exists
import os
if not os.path.exists('data/raw/NABIL_2000-01-01_2021-12-31.csv'):
    print("File not found!")

# Check data format
df = pd.read_csv('data/raw/NABIL_2000-01-01_2021-12-31.csv')
print(df.info())
print(df.head())
```

### Model Not Training:
```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.mean()}")
    else:
        print(f"{name}: No gradient")

# Check learning rate
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# Reduce learning rate if stuck
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0001
```

---

## 📚 Documentation Quick Links

- **README.md** - Project overview and setup
- **PYTORCH_GUIDE.md** - Complete implementation with code
- **PROJECT_STATUS.md** - Current progress and roadmap
- **QUICK_REFERENCE.md** - This file (common commands)

---

## 🎨 Visualization

### Plot Stock Price:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'])
plt.xlabel('Date')
plt.ylabel('Price (NPR)')
plt.title('Stock Price Over Time')
plt.grid(True)
plt.show()
```

### Plot Predictions:
```python
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Actual', alpha=0.7)
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
```

### Plot Training History:
```python
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🚀 Common Workflows

### 1. Quick Data Check:
```python
# One-liner to check a stock
import pandas as pd
df = pd.read_csv('data/raw/NABIL_2000-01-01_2021-12-31.csv')
print(f"Shape: {df.shape}, Date range: {df['Date'].min()} to {df['Date'].max()}")
```

### 2. Test GPU:
```python
# Quick GPU test
import torch
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f"✅ GPU computation successful! Result shape: {z.shape}")
```

### 3. Quick Model Test:
```python
# Test forward pass
model = StockLSTM(input_size=14).cuda()
dummy_input = torch.randn(32, 30, 14).cuda()  # (batch, seq_len, features)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be (32, 1)
```

---

## 📝 Notes

- Always use absolute paths for file operations
- Remember to move data to GPU with `.to(device)` or `.cuda()`
- Save models regularly during training
- Use `torch.no_grad()` for inference to save memory
- Clear GPU cache if memory issues occur

---

## 🔗 Useful Resources

- [PyTorch Docs](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LSTM Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

**Last Updated**: October 22, 2025
