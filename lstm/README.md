# 🚀 NEPSE Stock Price Prediction - LSTM with PyTorch
## Deep Learning Model for Nepal Stock Exchange Price Forecasting

This project implements an LSTM (Long Short-Term Memory) neural network using PyTorch to predict stock prices from the Nepal Stock Exchange (NEPSE). The model is trained on historical stock data from 2000-2021.

---

## 🎯 Project Overview

**Objective:** Build a robust LSTM model to predict stock closing prices based on historical trading data.

**Key Features:**
- PyTorch-based LSTM implementation with CUDA GPU acceleration
- Multi-stock dataset (400+ NEPSE stocks)
- Time series analysis and forecasting
- Feature engineering from OHLCV data
- Model training with GPU support (NVIDIA GTX 1650 Ti)

---

## �️ System Environment

**Hardware:**
- GPU: NVIDIA GeForce GTX 1650 Ti
- CUDA Version: 12.6
- Python: 3.13.2

**Software Stack:**
- PyTorch: 2.6.0+cu126 (CUDA enabled)
- CUDA available: ✅ Yes
- Parent directory Python environment: `C:/Users/Asus/AppData/Local/Programs/Python/Python313/python.exe`

**Note:** This LSTM project uses the PyTorch installation with CUDA support from the parent `torchproj` directory.

---

## 📁 Project Structure

```
lstm/
├── data/
│   ├── raw/                    # Raw NEPSE stock CSV files (400+ stocks, 2000-2021)
│   └── processed/              # Cleaned and engineered features
├── src/                        # Source code for models and utilities
├── models/                     # Saved model checkpoints
├── notebooks/                  # Jupyter notebooks for exploration
├── logs/                       # Training logs and metrics
└── README.md                   # This file
```

---

## 📊 Dataset Information

**Data Source:** Nepal Stock Exchange (NEPSE) historical data  
**Time Period:** 2000-01-01 to 2021-12-31  
**Number of Stocks:** 400+  
**File Format:** CSV

**Data Fields:**
- S.N.: Serial number
- Date: Trading date
- Total Transactions: Number of trades
- Total Traded Shares: Volume of shares
- Total Traded Amount: Total value in NPR
- Max. Price: Daily high
- Min. Price: Daily low
- Close Price: Closing price (target variable)

**Sample Data (NABIL):**
```
S.N.,Date,Total Transactions,Total Traded Shares,Total Traded Amount,Max. Price,Min. Price,Close Price
1,2021-12-29,3783,326139.00,477182107.70,1500.00,1450.00,1450.00
2,2021-12-28,1841,170269.00,253737724.20,1500.00,1479.20,1480.00
```

---

## 🏗️ Development Roadmap

### Phase 1: Data Exploration & Preprocessing ⏳
- [ ] Analyze raw NEPSE data structure
- [ ] Handle missing values and outliers
- [ ] Feature engineering (technical indicators, lag features)
- [ ] Normalize/standardize data
- [ ] Create train/validation/test splits
- [ ] Create PyTorch Dataset class

### Phase 2: LSTM Model Architecture 📐
- [ ] Design LSTM architecture (layers, hidden units, dropout)
- [ ] Implement custom PyTorch LSTM module
- [ ] Add attention mechanism (optional)
- [ ] Configure loss function and optimizer
- [ ] Setup learning rate scheduler

### Phase 3: Training Pipeline 🎯
- [ ] Create DataLoader with proper batching
- [ ] Implement training loop with GPU support
- [ ] Add validation metrics (MSE, MAE, RMSE)
- [ ] Implement early stopping
- [ ] Save model checkpoints
- [ ] Log training metrics (TensorBoard)

### Phase 4: Evaluation & Inference 📊
- [ ] Test on held-out data
- [ ] Visualize predictions vs actual
- [ ] Calculate performance metrics
- [ ] Create inference pipeline
- [ ] Multi-stock prediction capability

### Phase 5: Optimization & Deployment 🚀
- [ ] Hyperparameter tuning
- [ ] Model quantization for inference
- [ ] API endpoint (FastAPI)
- [ ] Documentation and examples

---

## � Quick Start

### Step 1: Verify Environment

Check CUDA availability:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Expected Output:**
```
PyTorch: 2.6.0+cu126
CUDA available: True
GPU: NVIDIA GeForce GTX 1650 Ti
```

### Step 2: Explore Raw Data

```python
import pandas as pd
import os

# List all stock files
raw_data_path = "data/raw"
stock_files = sorted([f for f in os.listdir(raw_data_path) if f.endswith('.csv')])
print(f"Total stocks: {len(stock_files)}")

# Load a sample stock
sample_df = pd.read_csv(os.path.join(raw_data_path, "NABIL_2000-01-01_2021-12-31.csv"))
print(sample_df.head())
print(f"\nDate range: {sample_df['Date'].min()} to {sample_df['Date'].max()}")
print(f"Total records: {len(sample_df)}")
```

### Step 3: Data Preprocessing (Next Steps)

Create a preprocessing script to:
1. Load all stock CSV files
2. Parse dates and sort chronologically
3. Handle missing values
4. Create technical indicators (MA, RSI, MACD)
5. Normalize features
6. Create sequences for LSTM input

---

## 📚 Project Tasks & Next Steps

### Immediate Tasks:
1. **Data Analysis** - Explore data quality, missing values, outliers
2. **Preprocessing Pipeline** - Clean and engineer features
3. **Dataset Class** - Create PyTorch Dataset for time series
4. **Model Architecture** - Design LSTM network
5. **Training Script** - Implement training loop with GPU

### Code Structure:
```
src/
├── data/
│   ├── __init__.py
│   ├── loader.py           # Data loading utilities
│   ├── preprocessor.py     # Cleaning and feature engineering
│   └── dataset.py          # PyTorch Dataset class
├── models/
│   ├── __init__.py
│   ├── lstm.py             # LSTM model architecture
│   └── attention.py        # Attention mechanism (optional)
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training loop
│   └── metrics.py          # Evaluation metrics
└── utils/
    ├── __init__.py
    ├── visualization.py    # Plotting utilities
    └── config.py           # Configuration management

1. Reads all CSV files from data/raw/
2. For each file:
   - Parse dates from DD/MM/YYYY to YYYY-MM-DD
   - Map columns:
     * Date → Date (converted format)
     * Min Price → Open (use min as approximation)
     * Max Price → High
     * Min Price → Low
     * Closing Price → Close
     * Total Traded Quantity → Volume
   - Remove currency symbols (₨, Rs.)
   - Remove comma separators in numbers
   - Sort by date ascending
   - Remove duplicates
   - Filter out rows with Close = 0 or Volume = 0
   - Keep only stocks with at least 365 days of data

3. Save processed files to data/processed/ as:
   stock_daily_<SYMBOL>.csv

4. Create a summary CSV with:
   - Stock symbol
   - Number of days
   - Date range
   - Average volume
   - Average price
   - Status (success/failed)

5. Save list of successfully processed stock symbols to:
   src/lstm_model/supported_stocks.txt

Include progress bar and error handling.
```

**Run the script:**
```bash
python src/convert_raw_to_processed.py
```

**Expected Output:**
```
✅ Processed 287 stocks successfully
📊 Summary saved to data/processed/conversion_summary.csv
📝 Supported stocks saved to src/lstm_model/supported_stocks.txt
```

---

## 🧠 Step 3: Build LSTM Model

### 3.1 Create Universal LSTM Training Script

**Copilot Prompt:**
```
Create a comprehensive LSTM training script src/train_universal_lstm_all.py that:

ARCHITECTURE:
- Multi-stock Universal LSTM model
- Uses stock embeddings (8-dimensional) to learn stock-specific patterns
- Dual input: time sequences (30 days) + stock ID
- 3 LSTM layers: 128 → 64 → 32 units
- Dropout layers (0.2) for regularization
- Predicts stock price 5 days ahead

DATA LOADING:
1. Read all stocks from data/processed/stock_daily_*.csv
2. For each stock:
   - Load OHLCV data
   - Create 14 features:
     * Basic: Open, High, Low, Close, Volume
     * Returns: return, volume_change
     * Moving averages: sma_5, sma_10, sma_20
     * Volatility: std_5, std_10, std_20
     * Momentum: rsi (14-period)
3. Create sequences:
   - Lookback window: 30 days
   - Forecast horizon: 5 days ahead
   - Label encode stock symbols
4. Combine all stocks into single dataset

PREPROCESSING:
- StandardScaler for features (fit on training data)
- StandardScaler for target (Close price)
- Train/test split: 80/20
- Save encoders and scalers as pickle files

MODEL TRAINING:
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Metrics: MAE
- Batch size: 256 (for GPU) or 64 (for CPU)
- Epochs: 150
- Callbacks:
  * ModelCheckpoint (save best model)
  * EarlyStopping (patience=15)
  * ReduceLROnPlateau (patience=5)

GPU OPTIMIZATION:
- Check for GPU availability
- Enable memory growth
- Use mixed precision (FP16) if GPU available
- Fall back to CPU with optimized settings

OUTPUT:
- Save model: src/lstm_model/universal_lstm_all_stocks.h5
- Save encoders: stock_encoder_all.pkl, scaler_X_all.pkl, scaler_y_all.pkl
- Print training progress with time estimates
- Show final metrics: Loss, MAE, R² score
- Save training history

Include detailed logging and error handling.
```

### 3.2 Create GPU Setup Guide

**Copilot Prompt:**
```
Create a GPU_SETUP_GUIDE.md that explains:

1. How to check if you have NVIDIA GPU:
   - Run nvidia-smi command
   - Check compute capability

2. TensorFlow GPU installation for Windows:
   - Python version requirements (3.10 recommended)
   - Install commands for tensorflow[and-cuda]
   - How to verify GPU detection

3. Troubleshooting common issues:
   - TensorFlow not detecting GPU
   - CUDA version mismatches
   - Memory errors

4. Alternative options:
   - Google Colab (free T4 GPU)
   - Cloud GPU providers (AWS, Google Cloud)
   - CPU optimization tips

5. Speed comparison table:
   - CPU vs different GPU types
   - Expected training times for our dataset

Include code snippets for testing GPU availability.
```

### 3.3 Train the Model

**Run training:**
```bash
python src/train_universal_lstm_all.py
```

**Expected Output:**
```
✅ GPU detected: NVIDIA GeForce GTX 1650 Ti
📊 Loading 287 stocks...
✅ Created 278,319 training sequences
🚀 Starting training...
Epoch 1/150: loss: 0.1234 - val_loss: 0.0987 - mae: 0.0543
...
✅ Training complete! Best R² score: 0.94
```

---

## 🔮 Step 4: Create Prediction System

### 4.1 Create Prediction Script

**Copilot Prompt:**
```
Create a comprehensive prediction script src/lstm_model/universal_predict_all.py that:

FUNCTIONALITY:
1. Load trained model and encoders
2. Support multiple prediction modes:
   - Single stock: --stock NABIL
   - Batch stocks: --batch NABIL SCB EBL
   - Top 20 stocks: --top20
   - All stocks: --all
   - List supported stocks: --list

PREDICTION PROCESS:
For each stock:
1. Load last 30 days of data from processed CSV
2. Calculate all 14 features
3. Scale features using saved scaler
4. Encode stock symbol
5. Make prediction
6. Inverse transform to get actual price
7. Calculate price change and percentage change

OUTPUT FORMAT:
- Current price
- Predicted price (5 days ahead)
- Expected change (NPR and %)
- Prediction date
- Trading signal:
  * 🟢 STRONG BUY (>5% gain)
  * 🟢 BUY (2-5% gain)
  * 🟡 HOLD (-2% to 2%)
  * 🔴 SELL (-5% to -2%)
  * 🔴 STRONG SELL (<-5%)

BATCH PREDICTIONS:
- Show top N gainers and losers
- Save all predictions to CSV
- Display formatted table
- Include error handling for missing data

Include command-line argument parsing and help messages.
```

### 4.2 Create Evaluation Script

**Copilot Prompt:**
```
Create an evaluation script src/evaluate_universal_lstm_all.py that:

1. Loads the trained model and test data
2. Makes predictions on test set
3. Calculates metrics:
   - R² score (overall and per-stock)
   - RMSE
   - MAE
   - MAPE (Mean Absolute Percentage Error)
4. Shows performance breakdown:
   - Best performing stocks
   - Worst performing stocks
   - Average metrics across all stocks
5. Creates visualizations:
   - Actual vs Predicted scatter plot
   - Error distribution histogram
   - Per-stock R² bar chart
6. Saves results to CSV

Include statistical analysis and confidence intervals.
```

---

## 📚 Step 5: Documentation

### 5.1 Create Comprehensive Documentation

**Copilot Prompt:**
```
Create ALL_STOCKS_DOCUMENTATION.md that includes:

1. Project Overview
   - What the model does
   - Dataset description (287 stocks, date range, etc.)
   - Model architecture details

2. Technical Specifications
   - Input features (14 features explained)
   - Model architecture (layers, parameters)
   - Training configuration
   - Performance metrics

3. Dataset Statistics
   - Total stocks: 287
   - Total trading days
   - Date range
   - Top 20 stocks by data size (table)

4. Usage Instructions
   - Training: python src/train_universal_lstm_all.py
   - Prediction: python src/lstm_model/universal_predict_all.py --stock NABIL
   - Evaluation: python src/evaluate_universal_lstm_all.py

5. Model Performance
   - R² score
   - RMSE, MAE, MAPE
   - Interpretation of metrics

6. File Structure
   - Where models are saved
   - Where predictions are saved
   - Where encoders/scalers are saved

Include tables, code examples, and expected outputs.
```

### 5.2 Create Quick Reference

**Copilot Prompt:**
```
Create QUICK_REFERENCE.md with:

1. Common Commands (copy-paste ready)
   - Train model
   - Predict single stock
   - Predict batch
   - Predict all stocks
   - List supported stocks

2. File Locations
   - Raw data
   - Processed data
   - Trained model
   - Predictions output

3. Troubleshooting
   - Model not found
   - Stock not supported
   - Insufficient data
   - GPU not detected

4. Quick Stats
   - Number of stocks
   - Model accuracy
   - Training time
   - Prediction time

Keep it concise and action-oriented.
```

### 5.3 Create Adding New Stocks Guide

**Copilot Prompt:**
```
Create ADDING_STOCKS.md that explains:

1. Prerequisites
   - Data format requirements
   - Minimum data requirements (365 days)

2. Step-by-step process:
   Step 1: Add raw CSV to data/raw/<SYMBOL>.csv
   Step 2: Run conversion script
   Step 3: Verify in supported_stocks.txt
   Step 4: Retrain model (or use existing for similar stocks)
   Step 5: Test prediction

3. Data validation
   - How to check if data is correctly formatted
   - Common errors and fixes

4. Retraining considerations
   - When to retrain
   - How long it takes
   - Performance impact

Include code examples and command-line instructions.
```

### 5.4 Create Project Status Report

**Copilot Prompt:**
```
Create PROJECT_STATUS.md that documents:

1. Current Status
   - Model version
   - Training status (completed/in-progress)
   - Number of stocks supported
   - Last update date

2. Performance Metrics
   - Overall R² score
   - RMSE, MAE, MAPE
   - Best/worst performing stocks

3. Known Issues
   - Any stocks with poor predictions
   - Data quality concerns
   - Model limitations

4. Risk Assessment
   - Model reliability for trading
   - Recommended risk management
   - Disclaimer about financial risks

5. Future Improvements
   - Potential enhancements
   - Additional features to add
   - Model architecture improvements

6. Changelog
   - Version history
   - Major updates
   - Bug fixes

Include tables and visual indicators (✅ ⚠️ ❌).
```

---

## 🎯 Step 6: Testing & Validation

### 6.1 Test Single Stock Prediction

**Run:**
```bash
python src/lstm_model/universal_predict_all.py --stock NABIL
```

**Expected Output:**
```
================================================================================
PREDICTION FOR NABIL
================================================================================
📅 Last Date:              2021-12-29
💰 Last Close Price:       NPR 1450.00
🔮 Predicted Price (5d):   NPR 1485.30
📊 Expected Change:        NPR +35.30 (+2.43%)
📅 Prediction Date:        ~2022-01-07
🎯 Signal:                 🟢 BUY
================================================================================
```

### 6.2 Test Batch Prediction

**Run:**
```bash
python src/lstm_model/universal_predict_all.py --top20
```

**Expected Output:**
```
🔮 Predicting for 20 stocks...
✅ [1/20] NABIL: +2.43%
✅ [2/20] SCB: +1.87%
...

================================================================================
TOP 10 POTENTIAL GAINERS (Next 5 Days)
================================================================================
🟢 NABIL    | Last: NPR  1450.00 | Predicted: NPR  1485.30 | Change: +2.43%
...
```

### 6.3 Evaluate Model Performance

**Run:**
```bash
python src/evaluate_universal_lstm_all.py
```

**Expected Output:**
```
================================================================================
OVERALL MODEL PERFORMANCE
================================================================================
R² Score:     0.9420
RMSE:         125.92 NPR
MAE:          94.63 NPR
MAPE:         12.71%

Top 5 Best Performing Stocks:
  NABIL: R² = 0.96
  SCB: R² = 0.94
  ...
```

---

## 📦 Step 7: Organize Project

### 7.1 Archive Old Experiments

**Copilot Prompt:**
```
Create an archive/ folder structure that organizes:

1. All non-LSTM experiments:
   - Random Forest models
   - LightGBM models
   - Ensemble approaches
   - Example pipelines

2. Maintain hierarchical structure:
   archive/
   └── src/
       ├── example-pipeline.py
       ├── pipeline-nabil.py
       └── random_forest_model/
           └── (all RF files)

3. Keep only LSTM-related files in main src/

4. Update README to reflect clean structure

5. Add ARCHIVE_README.md explaining what's archived and why
```

### 7.2 Create Documentation Index

**Copilot Prompt:**
```
Create DOCUMENTATION_INDEX.md that:

1. Lists all documentation files with descriptions
2. Organizes by use case:
   - Getting Started
   - Training
   - Prediction
   - Technical Details
   - Troubleshooting

3. Provides quick navigation links
4. Shows priority/importance levels
5. Indicates last update dates

Format as a searchable index with emojis for visual scanning.
```

---

## ✅ Final Checklist

After completing all steps, verify:

- [ ] Project structure is clean and organized
- [ ] All 287 stocks are processed and in `data/processed/`
- [ ] Model is trained and saved in `src/lstm_model/`
- [ ] Encoders and scalers are saved
- [ ] Prediction script works for single/batch/all stocks
- [ ] Evaluation script shows good metrics (R² > 0.85)
- [ ] All documentation is complete and accurate
- [ ] GPU is utilized if available
- [ ] Old experiments are archived
- [ ] README is updated with project overview

---

## 🚀 Quick Start Summary

**Minimum viable workflow:**

```bash
# Step 1: Setup
mkdir price-prediction && cd price-prediction
mkdir -p data/raw data/processed src/lstm_model

# Step 2: Add your raw CSV files to data/raw/

# Step 3: Convert data
python src/convert_raw_to_processed.py

# Step 4: Train model
python src/train_universal_lstm_all.py

# Step 5: Make predictions
python src/lstm_model/universal_predict_all.py --stock NABIL

# Step 6: Evaluate
python src/evaluate_universal_lstm_all.py
```

**Total time estimate:**
- Data conversion: 10-15 minutes
- Model training (GPU): 5-10 hours
- Model training (CPU): 3-5 days
- Documentation: 1-2 hours

---

## 📞 Support & Troubleshooting

**Common Issues:**

1. **GPU not detected**
   - See GPU_SETUP_GUIDE.md
   - Consider Google Colab alternative

2. **Stock data format errors**
   - Check DATA_FORMAT_SPECIFICATION.md
   - Verify column names and date format

3. **Model accuracy too low**
   - Check if data has quality issues
   - Verify sufficient training data (>365 days per stock)
   - Consider feature engineering improvements

4. **Prediction errors**
   - Ensure stock exists in supported_stocks.txt
   - Check if processed CSV file exists
   - Verify model and encoders are loaded correctly

---

## 🎓 Learning Resources

**To understand the code better:**

1. LSTM basics: Understanding sequential models
2. Time series forecasting: Walk-forward validation
3. Feature engineering: Technical indicators (SMA, RSI, etc.)
4. Model evaluation: R², RMSE, MAE, MAPE metrics
5. Risk management: Position sizing, stop-loss strategies

**Recommended next steps:**

1. Add more technical indicators (MACD, Bollinger Bands)
2. Implement ensemble with other models
3. Add news sentiment analysis
4. Create web dashboard for predictions
5. Implement automated trading signals

---

## ⚠️ Important Disclaimers

**This model is for educational purposes:**

- **NOT financial advice** - Do not trade solely based on predictions
- **Past performance ≠ future results** - Markets can be unpredictable
- **Use proper risk management** - Never risk more than you can afford to lose
- **Backtesting required** - Test strategies before real trading
- **Market conditions change** - Model may need retraining
- **Regulatory compliance** - Follow local trading regulations

**Risk factors:**

- Model can't predict black swan events
- News and fundamentals not incorporated
- 5-day predictions have uncertainty
- Some stocks perform better than others
- Market manipulation can distort predictions

---

## 📄 License & Credits

**This project uses:**

- TensorFlow/Keras (Apache 2.0)
- scikit-learn (BSD-3-Clause)
- pandas (BSD-3-Clause)
- NumPy (BSD-3-Clause)

**Data source:** NEPSE (Nepal Stock Exchange)

**Created by:** [Your Name/Team]  
**Last Updated:** [Date]  
**Version:** 1.0

---

**End of Recreation Guide** 🎉

Follow these steps in order, and you'll have a fully functional Universal LSTM stock price prediction system!
