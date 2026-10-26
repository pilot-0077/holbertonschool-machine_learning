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
- `score_news.py`: creates baseline sentiment scores from timestamped headlines
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

Download one-minute Bitcoin data from Coinbase and/or Bitstamp. The script supports both the original Kaggle-style columns (such as `Timestamp`, `Volume_(BTC)`, and `Volume_(Currency)`) and simpler OHLCV CSVs with lowercase columns (`timestamp,open,high,low,close,volume`).

One public Bitstamp historical dataset is available in the [Bitstamp BTC/USD minute-data repository](https://github.com/ff137/bitstamp-btcusd-minute-data). Its historical file is compressed, so you can pass the `.csv.gz` file directly to pandas. The repository documents the source and applicable data licence. The historical bulk data is derived from the Kaggle dataset by Zielak (mczielinski) and is published under CC BY-SA 4.0; retain the required attribution if you use or redistribute it.

When a source file contains only BTC volume (the lowercase `volume` column), the preprocessing script estimates currency volume as BTC volume multiplied by the closing price. Treat this as an approximation, not as an exchange-reported currency-volume measurement.

Run preprocessing with the actual filename you downloaded. For example:

```bash
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --output btc_market_only.npz
```

If you downloaded the original Kaggle files instead, pass the relevant filenames. If you have only one exchange file, pass only that filename.

On Windows, you can run the full market-only workflow with the helper script after placing the dataset in this directory:

```powershell
.\run_market_experiment.ps1
```

If the dataset is stored elsewhere, provide its path:

```powershell
.\run_market_experiment.ps1 -DataFile "C:\path\to\btcusd_bitstamp_1min_2012-2025.csv.gz"
```

The helper creates a virtual environment if needed, installs dependencies, preprocesses the data, and trains/evaluates the model. The first installation may take several minutes.

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

The code can also use news headlines if you have a CSV containing publication timestamps and headline text. First, score the headlines with the included VADER-based baseline:

```bash
python score_news.py news_headlines.csv --output news_sentiment.csv
```

Then prepare and train the news-enhanced model:

```bash
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --news-file news_sentiment.csv --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news --epochs 50 --batch-size 64
```

Read [NEWS_DATA_GUIDE.md](NEWS_DATA_GUIDE.md) for the required input columns and precautions. VADER is a general-purpose sentiment baseline, not a finance-specific model. Use real headlines with accurate publication times, and do not invent news data.

Compare the two experiments using the same date range and split strategy:

- Market-only: price and volume features
- Market plus news: the same features, with hourly sentiment and article count

Only conclude that news improves forecasting if the held-out test metrics support that conclusion. News data must be timestamped carefully to prevent future information from leaking into the input windows.

## Important limitations

Historical price and volume data do not capture every factor affecting Bitcoin. Even a model that beats a simple baseline on one test period may fail in a different market regime. This project is for learning and experimentation, not financial advice.
