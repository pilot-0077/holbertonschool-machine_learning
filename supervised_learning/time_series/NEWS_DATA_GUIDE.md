# Optional News Sentiment Data

The forecasting project can compare a market-only model with a model that also receives hourly news-sentiment features. This extension requires a news dataset; it is not included in the repository.

## Expected CSV format

Create a CSV file named, for example, `news_sentiment.csv`, with one row per article and these columns:

```csv
Timestamp,sentiment_score
2024-01-15 09:12:00,0.65
2024-01-15 09:47:00,-0.30
2024-01-15 10:05:00,0.20
```

The rows above are illustrative examples only. Replace them with real articles and real sentiment scores before training.

- `Timestamp`: article publication time in UTC. Unix timestamps in seconds are also accepted.
- `sentiment_score`: numeric sentiment score produced by a documented sentiment-analysis method. A common convention is -1 for negative, 0 for neutral, and +1 for positive.

The preprocessing script calculates the average sentiment and article count for each hour. Hours without news receive a sentiment score of 0 and a count of 0. It then normalizes these features using training-period statistics.

## Avoiding data leakage

The publication time matters. Only use news that would have been available at the time of a forecast. Do not use a later article to explain an earlier price, and do not assign a sentiment score derived from future information to an earlier timestamp.

For a rigorous experiment, record the news source, publication timestamp, sentiment model or labeling method, and any filtering rules. If articles are duplicated across sources, consider deduplicating them.

## Running the news-enhanced experiment

```bash
python preprocess_data.py bitstampUSD_1-min_data.csv coinbaseUSD_1-min_data.csv --news-file news_sentiment.csv --output btc_with_news.npz
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news
```

Run the market-only experiment on the same market dates, then compare the held-out test MAE and RMSE. The news model is useful only if it improves performance reliably; extra features do not guarantee a better forecast.

This repository prepares the model input from pre-scored news data. It does not automatically collect news or generate sentiment scores.
