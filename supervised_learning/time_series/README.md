# Time Series Forecasting: Bitcoin

This project explores Bitcoin time-series forecasting with TensorFlow. It uses the previous 24 hours of market observations to estimate the next-hour log return, then converts that estimate into a closing-price forecast.

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
- `run_market_experiment.ps1`: Windows helper to run the market-only experiment
- `run_news_experiment.ps1`: Windows helper to run the same experiment with news sentiment
- `BLOG_POST_DRAFT.md` and `bitcoin_forecasting_cover.svg`: blog draft and cover image

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
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --start-date "2017-01-01" --end-date "2025-01-07 00:00:00" --output btc_market_only_returns.npz
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

The helper creates a virtual environment if needed, installs dependencies, preprocesses the data, and trains/evaluates the return-based model. It writes to `btc_market_only_returns.npz`, `models/btc_market_only_returns.keras`, and `results/market_only_returns` so results from the earlier raw-price experiment are preserved. The first installation may take several minutes.

The script aggregates minute-level rows into hourly observations, handles missing values, combines the supplied exchange data, and scales features using statistics from the training portion only. The data is split chronologically:

- First 70%: training
- Next 15%: validation and early stopping
- Final 15%: held-out test set

The test set is not used to select the best model.

## 3. Train and evaluate the market-only model

```bash
python forecast_btc.py --data btc_market_only_returns.npz --model models/btc_market_only_returns.keras --plot-dir results/market_only_returns --epochs 50 --batch-size 64
```

The GRU model uses 24 hourly observations, two GRU layers (64 and 32 units), a dense layer with 16 neurons, and a one-value output layer. Instead of predicting the raw price level, it predicts the next-hour log return. Price-relative features and log-transformed volume features help reduce scale changes across market regimes. The predicted return is converted back into a USD closing-price estimate for evaluation. Training-only statistics are used to scale features and the target. The model uses Adam, mean squared error loss, early stopping, and model checkpointing.

The script reports validation and test errors on normalized log returns. It also converts forecasts back into USD closing prices and compares test MAE and RMSE with a naive baseline: predicting that the next closing price will equal the most recent closing price.

The output directory contains:

- `training_validation_loss.png`: training and validation loss by epoch
- `actual_vs_predicted_test.png`: actual test prices, GRU predictions, and baseline
- `test_predictions.csv`: timestamped actual values and predictions

These files are generated from the actual run. Use their measured metrics in any report or blog post.

## Observed market-only results

The market-only experiment was run on the selected historical period from 2017-01-01 through 2025-01-07. The raw-price GRU and return-based GRU were evaluated against a previous-close baseline on the held-out test split.

| Test metric | Raw-price GRU | Return-based GRU | Previous-close baseline |
|---|---:|---:|---:|
| MAE in USD | $1,827.48 | $224.45 | $224.78 |
| RMSE in USD | $4,181.04 | $355.06 | $355.25 |

The return-based GRU's normalized test MSE was 0.351838 and normalized test MAE was 0.389412. Those normalized metrics apply to the log-return target and are not directly comparable with normalized errors from the raw-price model.

The return-based GRU's USD errors were only marginally lower than the baseline (MAE by $0.33 and RMSE by $0.19). Treat the model as an educational experiment, not as evidence of a meaningful forecasting advantage.

The news-enhanced experiment has now been run on the same 70,272 hourly observations and chronological split. The CSV's `sentiment_label` supplies direction and `sentiment_score` supplies confidence; preprocessing converts these to signed sentiment before hourly aggregation.

| Test metric | Market-only GRU | News-enhanced GRU | Previous-close baseline |
|---|---:|---:|---:|
| MAE in USD | $224.45 | $225.05 | $224.78 |
| RMSE in USD | $355.06 | $355.58 | $355.25 |

The news-enhanced model's normalized test MSE was 0.352685 and normalized test MAE was 0.390441. It did not improve on the market-only GRU or the naive baseline in this run. The differences are small, so this is not proof that news sentiment can never help; it means an advantage was not demonstrated with this dataset, feature construction, and model.

## Experiment graphs

The repository includes four SVG redraws based on the locally generated experiment plots. The original PNGs are retained in the local `results/market_only_returns` and `results/with_news_returns` folders.

- [Market-only training and validation loss](market_only_returns_training_validation_loss.svg)
- [Market-only actual vs. predicted prices](market_only_returns_actual_vs_predicted_test.svg)
- [News-enhanced training and validation loss](with_news_returns_training_validation_loss.svg)
- [News-enhanced actual vs. predicted prices](with_news_returns_actual_vs_predicted_test.svg)

## 4. Optional experiment: add news sentiment

For the news extension, you can use a pre-scored news dataset or score your own timestamped headlines. The [CryptoVision dataset on Mendeley Data](https://data.mendeley.com/datasets/3c3xtxtfb6/2) is one possible research source; its published schema includes timestamped sentiment scores and a coin type. Read [NEWS_DATA_GUIDE.md](NEWS_DATA_GUIDE.md) for download, licence, date alignment, and limitations.

The CryptoVision CSV used in this experiment has `Date Time`, `Coin Type`, `sentiment_label`, and `sentiment_score` columns. The preprocessing code normalizes header case and spaces, filters Bitcoin/BTC rows, and combines the sentiment label with the confidence score to form a signed sentiment feature. It uses the timestamp, coin, label, and score rather than the dataset's market columns:

```bash
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --news-file cryptovision.csv --start-date "2017-01-01" --end-date "2025-01-07 00:00:00" --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news --epochs 50 --batch-size 64
```

If you have raw headlines instead, score them first and use the same date range:

```bash
python score_news.py news_headlines.csv --output news_sentiment.csv
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --news-file news_sentiment.csv --start-date "2017-01-01" --end-date "2025-01-07 00:00:00" --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news --epochs 50 --batch-size 64
```

VADER is a general-purpose baseline, not a finance-specific model. Use real headlines with accurate publication times, respect dataset terms, and do not invent news data.

Compare the two experiments using the same date range and split strategy:

- Market-only: price and volume features
- Market plus news: the same features, with hourly sentiment and article count

Only conclude that news improves forecasting if the held-out test metrics support that conclusion. News data must be timestamped carefully to prevent future information from leaking into the input windows.

## Important limitations

Historical price and volume data do not capture every factor affecting Bitcoin. Even a model that beats a simple baseline on one test period may fail in a different market regime. This project is for learning and experimentation, not financial advice.
