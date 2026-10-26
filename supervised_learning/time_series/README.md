# Time Series Forecasting: Bitcoin

This project explores Bitcoin time-series forecasting with TensorFlow. It uses the previous 24 hours of market observations to estimate the next hourly closing price.

The project is an educational experiment. It does not promise accurate forecasts or investment returns.

## What you will learn

- How to aggregate minute-level market data into hourly features
- Why feature scaling must be fitted on training data only
- How to build sliding windows and a TensorFlow data pipeline
- How to train a GRU recurrent neural network
- Why chronological train, validation, and test splits matter
- How to compare a model with a simple baseline
- How optional hourly news-sentiment features can be evaluated

## Project files

- `preprocess_data.py`: prepares market data and optional news features
- `forecast_btc.py`: trains the GRU model, evaluates it, and saves plots
- `requirements.txt`: Python dependencies
- `NEWS_DATA_GUIDE.md`: format and precautions for optional news data

## 1. Set up the environment

Use a Python version supported by the TensorFlow version installed on your system.

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

## 2. Prepare market data

Download the one-minute Bitcoin CSV file or files from Coinbase and/or Bitstamp. Each file must include the columns expected by the script, including `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume_(BTC)`, `Volume_(Currency)`, and `Weighted_Price`.

Run preprocessing with the actual filenames you downloaded. For example:

```bash
python preprocess_data.py bitstampUSD_1-min_data.csv coinbaseUSD_1-min_data.csv --output btc_market_only.npz
```

If you have only one exchange file, pass only that filename.

The script aggregates minute-level rows into hourly observations, handles missing values, combines the supplied exchange data, and scales features using statistics from the training portion only. The data is split chronologically:

- First 70%: training
- Next 15%: validation and early stopping
- Final 15%: held-out test set

The test set is not used to select the best model.

## 3. Train and evaluate the market-only model

```bash
python forecast_btc.py --data btc_market_only.npz --model models/btc_market_only.keras --plot-dir results/market_only --epochs 50 --batch-size 64
```

The GRU model uses 24 hourly observations, two GRU layers (64 and 32 units), a dense layer with 16 neurons, and a one-value output layer. It uses the Adam optimizer, mean squared error loss, early stopping, and model checkpointing.

The script reports validation and test errors. It also converts test MAE and RMSE back to approximate USD units and compares the model with a naive baseline: predicting that the next closing price will equal the most recent closing price.

The output directory contains:

- `training_validation_loss.png`: training and validation loss by epoch
- `actual_vs_predicted_test.png`: actual test prices, GRU predictions, and baseline
- `test_predictions.csv`: timestamped actual values and predictions

These files are generated from the actual run. Use their measured metrics in any report or blog post.

## 4. Optional experiment: add news sentiment

The code can also use a CSV containing a timestamp and a sentiment score for each news article. Read [NEWS_DATA_GUIDE.md](NEWS_DATA_GUIDE.md) before creating the file. The sentiment scores must come from a real, documented source or a sentiment-analysis process, not invented example values.

Example command:

```bash
python preprocess_data.py bitstampUSD_1-min_data.csv coinbaseUSD_1-min_data.csv --news-file news_sentiment.csv --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news --epochs 50 --batch-size 64
```

Compare the two experiments using the same date range and split strategy:

- Market-only: price and volume features
- Market plus news: the same features, with hourly sentiment and article count

Only conclude that news improves forecasting if the held-out test metrics support that conclusion. News data must be timestamped carefully to prevent future information from leaking into the input windows.

## Important limitations

Historical price and volume data do not capture every factor affecting Bitcoin. Even a model that beats a simple baseline on one test period may fail in a different market regime. This project is for learning and experimentation, not financial advice.
