# Time Series Forecasting

This project explores time-series forecasting with TensorFlow. The goal is to use the previous 24 hours of Bitcoin market data to estimate the next hour's closing price.

## What you will learn

- What time-series forecasting and sliding windows are
- How to clean and aggregate minute-level market data
- Why normalization must use training data statistics only
- How to build a TensorFlow data pipeline
- How to train and evaluate a GRU-based recurrent neural network
- How to compare model predictions with a simple baseline

## Project files

- `preprocess_data.py`: prepares raw exchange CSV files and creates `btc_hourly.npz`
- `forecast_btc.py`: trains the model, evaluates it, and saves two plots
- `requirements.txt`: lists the Python dependencies

## 1. Set up the environment

Use a Python version supported by your installed TensorFlow release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Prepare the raw data

Download the one-minute Bitcoin CSV dataset files from Coinbase and/or Bitstamp. The files must contain the columns expected by the preprocessing script, including `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume_(BTC)`, and `Volume_(Currency)`.

Run the script with the actual filenames you downloaded:

```bash
python preprocess_data.py bitstampUSD_1-min_data.csv coinbaseUSD_1-min_data.csv
```

Use only the file or files you have. For example, if you have only the Bitstamp file:

```bash
python preprocess_data.py bitstampUSD_1-min_data.csv
```

The script aggregates minute-level observations into hourly rows, handles missing values, combines supplied exchange data, and normalizes features using statistics calculated from the first 80% of the chronological data. The remaining 20% is held out for validation.

The output is `btc_hourly.npz`. The raw CSV files are not included in this repository; provide their local paths when running preprocessing.

## 3. Train and evaluate the model

```bash
python forecast_btc.py --epochs 50 --batch-size 64
```

Optional arguments:

- `--data`: path to the preprocessed NPZ file
- `--model`: path for the saved model
- `--plot-dir`: directory for evaluation plots
- `--epochs`: maximum number of training epochs
- `--batch-size`: training batch size

The model uses 24 consecutive hourly observations to predict the next hour's closing price. It contains two GRU layers (64 and 32 units), followed by a dense layer with 16 neurons and a single output neuron. Training uses Adam, MSE loss, early stopping, and model checkpointing.

The script reports normalized validation MSE and MAE, converts MAE and RMSE to approximate USD errors, and compares the model with a naive baseline that predicts the next close will equal the most recent close.

## 4. View the generated plots

After training, the `plots/` directory contains:

- `training_validation_loss.png`: training loss and validation loss across epochs
- `actual_vs_predicted.png`: actual validation prices, GRU predictions, and the naive baseline

These figures are generated from the actual training run. Do not publish example or illustrative charts as if they were measured model results.

## Important limitations

This is an educational experiment, not financial advice. Historical prices and trading volumes do not capture every event that affects Bitcoin. Forecasting performance should be evaluated against a baseline and across different market periods before drawing conclusions.
