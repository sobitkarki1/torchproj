# 🚀 Python Scripts Workflow Guide

## Quick Start - Run Everything

The easiest way to run the complete pipeline:

```bash
cd c:\Users\Asus\Development\torchproj\lstm
python run_pipeline.py
```

This will execute all steps automatically with prompts between each stage.

---

## Individual Script Usage

### 1. Data Exploration 📊

Analyze stock data and generate visualizations:

```bash
python src/data/explore.py
```

**Output:**
- `data/exploration_results/` - Directory with visualizations:
  - `NABIL_price_trend.png`
  - `NABIL_moving_averages.png`
  - `NABIL_ohlc_volume.png`
  - `NABIL_distributions.png`
  - `NABIL_correlation.png`
  - `stocks_comparison.png`

---

### 2. Data Preprocessing 🔧

Process raw data and add technical indicators:

```bash
python src/data/preprocessor.py
```

**Output:**
- `data/processed/NABIL_processed.csv` - Processed data with 24 features
- `models/scalers/` - Fitted StandardScalers for features and target

**Features Added:**
- Basic: Max Price, Min Price, Close Price, Volume, Amount
- Returns: return, volume_change
- Moving Averages: sma_5, sma_10, sma_20, ema_5, ema_10
- Volatility: std_5, std_10, std_20
- Indicators: RSI, MACD, Bollinger Bands, Momentum

---

### 3. Dataset Creation Test 📦

Test PyTorch Dataset creation:

```bash
python src/data/dataset.py
```

**Tests:**
- Sequence creation (30-day lookback)
- DataLoader batching
- GPU data transfer
- Sample retrieval

---

### 4. Model Architecture Test 🧠

Test LSTM model creation and forward pass:

```bash
python src/models/lstm.py
```

**Tests:**
- Basic LSTM model
- Attention LSTM model
- GPU compatibility
- Parameter counting

---

### 5. Model Training 🎓

Train LSTM model on stock data:

```bash
python src/training/trainer.py
```

**Configuration:**
- Sequence length: 30 days
- Prediction horizon: 5 days ahead
- Batch size: 256
- Learning rate: 0.001
- Early stopping patience: 15 epochs

**Output:**
- `models/checkpoints/best_model.pth` - Best model during training
- `models/lstm_final.pth` - Final trained model

**Training Features:**
- GPU acceleration (GTX 1650 Ti)
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Model checkpointing

---

## Complete Workflow

### Step-by-Step:

```bash
# 1. Explore data
python src/data/explore.py

# 2. Review visualizations
explorer data/exploration_results  # Windows Explorer

# 3. Preprocess data
python src/data/preprocessor.py

# 4. Test dataset
python src/data/dataset.py

# 5. Test model
python src/models/lstm.py

# 6. Train model
python src/training/trainer.py
```

### Or run everything:

```bash
python run_pipeline.py
```

---

## Script Details

### Data Exploration (`src/data/explore.py`)

Functions:
- `explore_available_data()` - List all stock files
- `load_and_analyze_stock()` - Load and analyze stock
- `calculate_technical_indicators()` - Add indicators
- `plot_price_trends()` - Generate visualizations
- `compare_multiple_stocks()` - Compare stocks

### Data Preprocessing (`src/data/preprocessor.py`)

Class: `StockPreprocessor`
- `add_technical_indicators()` - Feature engineering
- `select_features()` - Feature selection
- `fit_transform()` - Fit and scale data
- `save_scalers()` - Save for inference
- `load_scalers()` - Load saved scalers

### Dataset (`src/data/dataset.py`)

Class: `StockDataset`
- Creates sequences for LSTM input
- Handles train/val splitting
- PyTorch DataLoader integration
- GPU-compatible

### Model (`src/models/lstm.py`)

Classes:
- `StockLSTM` - Basic multi-layer LSTM
- `AttentionLSTM` - LSTM with attention mechanism
- `create_model()` - Factory function

### Training (`src/training/trainer.py`)

Class: `LSTMTrainer`
- Complete training loop
- Validation
- Learning rate scheduling
- Early stopping
- Checkpointing

---

## Configuration

Edit configuration in each script's `main()` function:

**Preprocessing:**
```python
# In src/data/preprocessor.py
scaling_method = 'standard'  # or 'minmax'
```

**Dataset:**
```python
# In src/data/dataset.py
SEQUENCE_LENGTH = 30  # lookback window
PREDICTION_HORIZON = 5  # days ahead
```

**Training:**
```python
# In src/training/trainer.py
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
```

---

## Expected Runtime

On GTX 1650 Ti GPU:
- Data Exploration: ~30 seconds
- Data Preprocessing: ~10 seconds
- Dataset Test: ~5 seconds
- Model Test: ~2 seconds
- Training (100 epochs): ~15-30 minutes

On CPU:
- Training: ~2-4 hours

---

## Troubleshooting

### Missing Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install pandas numpy matplotlib seaborn scikit-learn tqdm
```

### GPU Out of Memory

Reduce batch size in `src/training/trainer.py`:
```python
BATCH_SIZE = 128  # or 64
```

### File Not Found

Ensure you're in the `lstm/` directory:
```bash
cd c:\Users\Asus\Development\torchproj\lstm
```

### Import Errors

Make sure Python can find modules:
```bash
# Set PYTHONPATH if needed
set PYTHONPATH=%CD%
```

---

## Output Structure

After running all scripts:

```
lstm/
├── data/
│   ├── exploration_results/  ← Visualizations
│   └── processed/            ← Processed CSV
├── models/
│   ├── checkpoints/          ← Training checkpoints
│   ├── scalers/              ← Fitted scalers
│   ├── best_model.pth        ← Best model
│   └── lstm_final.pth        ← Final model
└── logs/                     ← (Future: TensorBoard logs)
```

---

## Next Steps After Training

1. **Evaluate Model**: Create evaluation script
2. **Make Predictions**: Create inference script
3. **Visualize Results**: Plot predictions vs actual
4. **Experiment**: Try different architectures
5. **Scale Up**: Process multiple stocks

---

## Tips

- **Start Simple**: Run each script individually to understand the workflow
- **Monitor GPU**: Use Task Manager to check GPU usage
- **Review Outputs**: Check generated files after each step
- **Experiment**: Modify configurations to see effects
- **Save Work**: Commit changes to git regularly

---

**Last Updated**: October 22, 2025
