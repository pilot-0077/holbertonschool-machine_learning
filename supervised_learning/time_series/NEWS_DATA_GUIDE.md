# Optional News Sentiment Data

This extension compares a market-only model with a model that also receives hourly news-sentiment features. The repository does not include a news corpus; you must obtain a dataset and check its usage terms.

## Option A: Use a pre-scored news dataset

One possible research source is the [CryptoVision cryptocurrency news dataset on Mendeley Data](https://data.mendeley.com/datasets/wvjjxr8bxx/2). Its dataset description lists fields including `Date_Time`, `Title`, `Coin_Type`, and `Sentiment_Score`, with cryptocurrency news from 2017–2025.

After downloading the data, save the relevant CSV locally. The preprocessing script accepts common timestamp and score names, including `Date_Time` and `Sentiment_Score`. If the file includes a coin/asset column, it filters for rows whose value contains Bitcoin or BTC.

Use the same overlapping date range for both experiments. The historical Bitstamp market file in the README covers through 7 January 2025, so an example is:

```bash
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --start-date "2017-01-01" --end-date "2025-01-07 00:00:00" --output btc_market_only.npz
python forecast_btc.py --data btc_market_only.npz --model models/btc_market_only.keras --plot-dir results/market_only

python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --news-file cryptovision.csv --start-date "2017-01-01" --end-date "2025-01-07 00:00:00" --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news
```

Replace `cryptovision.csv` with the filename you downloaded. The script aggregates article scores into hourly average sentiment and article count. Hours without an article in the provided dataset receive a sentiment score of 0 and a count of 0.

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
