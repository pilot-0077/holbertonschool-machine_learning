# Time Series Forecasting

This project focuses on forecasting data that is ordered over time using
recurrent neural networks and TensorFlow data pipelines.

## Learning Objectives

By the end of this project, I should be able to explain:

- What time series forecasting is
- What a stationary process is
- What a sliding window is
- How to preprocess time series data
- How to create a TensorFlow data pipeline for time series data
- How to perform time series forecasting with RNNs in TensorFlow

## Task 0 - When to Invest

The goal is to forecast the Bitcoin closing price for the following hour
using the previous 24 hours of market data.

The raw Coinbase and Bitstamp datasets contain one-minute observations with:

- Unix timestamp
- Open, high, low, and close prices
- BTC transaction volume
- Currency transaction volume
- Volume-weighted average price

### `preprocess_data.py`

The preprocessing script:

- loads one or both raw exchange CSV files
- aggregates one-minute observations into hourly OHLCV data
- handles missing hourly intervals
- combines Coinbase and Bitstamp data when both are supplied
- keeps the useful market features
- performs a chronological train/validation split
- normalizes data using training statistics only
- saves the processed arrays to `btc_hourly.npz`

Example:

```bash
./preprocess_data.py bitstampUSD_1-min_data.csv \
    coinbaseUSD_1-min_data.csv
```

If no paths are supplied, the script searches the current directory for
common Coinbase and Bitstamp one-minute CSV filenames.

### `forecast_btc.py`

The forecasting script:

- loads `btc_hourly.npz`
- creates 24-hour sliding windows
- feeds the windows through a `tf.data.Dataset`
- trains a stacked GRU model
- uses mean-squared error as the loss function
- validates on chronologically later data
- saves the best model as `btc_forecaster.keras`

Run it with:

```bash
./forecast_btc.py
```

Optional arguments include `--epochs`, `--batch-size`, `--data`, and
`--model`.

## Files

- `README.md`
- `preprocess_data.py`
- `forecast_btc.py`

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/time_series`
