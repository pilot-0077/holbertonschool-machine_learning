# Optional News Sentiment Data

This extension compares a market-only model with a model that also receives hourly news-sentiment features. The repository does not include a news corpus; you must obtain a dataset and check its usage terms.

## Option A: Use a pre-scored news dataset

Use the [CryptoVision cryptocurrency news dataset on Mendeley Data](https://data.mendeley.com/datasets/3c3xtxtfb6/2). Its published schema includes `Date_Time`, `Title`, `Coin_Type`, and `Sentiment_Score`, with cryptocurrency news from 2017–2025. The dataset is listed under CC BY 4.0; cite the authors and check the licence before redistributing any data. Download the dataset and extract the CSV file locally; do not commit the full news dataset to this repository.

After downloading the data, save the relevant CSV locally. Header names may vary between releases: the CSV used in our test has `Date Time`, `Coin Type`, `sentiment_label`, and `sentiment_score`. The preprocessing script normalizes case, spaces, and hyphens in column names, then filters the dataset to Bitcoin/BTC rows. The dataset documentation describes `Sentiment_Score` as model confidence, not sentiment direction. When a sentiment label is present, the script converts confidence to signed sentiment: positive = +confidence, negative = -confidence, and neutral = 0. It uses the publication timestamp, coin, label, and score only; it does not use the CSV's OHLCV or `Market_Move` columns as model features.

Use the same overlapping date range for both experiments. The historical Bitstamp market file covers through 7 January 2025, so the helper uses that end date by default.

```powershell
.\run_news_experiment.ps1 -DataFile "D:\path\to\btcusd_bitstamp_1min_2012-2025.csv" -NewsFile ".\cryptovision.csv"
```

The helper uses the same 2017-01-01 through 2025-01-07 date window and return-based target as the market-only experiment. It writes separate outputs to `btc_with_news_returns.npz`, `models/btc_with_news_returns.keras`, and `results/with_news_returns` so it does not overwrite the market-only run.

Replace `cryptovision.csv` with the filename you downloaded. The preprocessing script aggregates the signed article sentiment scores into hourly average sentiment and article count. Hours without a matching article in the provided dataset receive a sentiment score of 0 and a count of 0.

Check the news dataset's licence and citation requirements before using or redistributing it. A news corpus may not contain every article published during each hour, so a zero count means no matching row in that dataset, not proof that no news existed.

## Option B: Score your own headline dataset

If your CSV contains headlines but no sentiment score, prepare a file such as `news_headlines.csv` with a publication timestamp and headline/title/text column:

```csv
Timestamp,headline
2024-01-15 09:12:00+00:00,"Example headline about a Bitcoin market update"
2024-01-15 09:47:00+00:00,"Example headline about cryptocurrency regulation"
```

These rows are illustrative only, not real historical news. Replace them with real articles from a source whose use is permitted.

Run the included VADER baseline scorer:

```bash
python score_news.py news_headlines.csv --output news_sentiment.csv
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --news-file news_sentiment.csv --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news
```

VADER's compound score ranges from -1 (negative) to +1 (positive). It is a general-purpose baseline, not a finance-specific model. It may misunderstand financial terminology and cannot reliably tell whether a positive-sounding headline will cause the price to rise.

## Avoiding data leakage

- Use publication timestamps, not a later crawl time.
- Do not use articles published after the forecast target time.
- Use the same market date range and train/validation/test split for both models.
- Do not use future market movements to assign earlier sentiment labels.
- Document the source, date range, filtering, duplicate handling, and scoring method.
- Compare held-out test MAE and RMSE; do not assume extra features improve the model.

This repository provides a baseline for scoring and joining headlines. It does not automatically collect news or guarantee that news sentiment improves Bitcoin forecasts.
