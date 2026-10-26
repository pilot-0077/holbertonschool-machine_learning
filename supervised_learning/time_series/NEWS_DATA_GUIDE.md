# Optional News Sentiment Data

This extension compares a market-only model with a model that also receives hourly news-sentiment features. It requires a timestamped news-headline dataset; no news corpus is included in this repository.

## 1. Prepare the headline CSV

Create a file such as `news_headlines.csv` with one row per article. It must contain a publication timestamp and a headline/title/text column.

```csv
Timestamp,headline
2024-01-15 09:12:00+00:00,"Example headline about a Bitcoin market update"
2024-01-15 09:47:00+00:00,"Example headline about regulation and cryptocurrency"
2024-01-15 10:05:00+00:00,"Example headline about market volatility"
```

These rows are illustrative only, not real historical news and not suitable for model training. Replace them with real articles from a source whose use is permitted.

The timestamp column can be named `Timestamp`, `timestamp`, `published_at`, `publishedAt`, `date`, or `time`. It must represent the article's publication time in UTC; numeric Unix timestamps in seconds are also accepted. The headline column can be named `headline`, `title`, `article_title`, or `text`.

## 2. Score the headlines

Run the included script:

```bash
python score_news.py news_headlines.csv --output news_sentiment.csv
```

The script uses VADER's compound sentiment score, which ranges from -1 (negative) to +1 (positive). VADER is a general-purpose baseline, not a finance-specific sentiment model. Sarcasm, financial terminology, and the difference between a positive headline and a positive price impact may be missed.

The output contains `Timestamp` and `sentiment_score` columns, one row per valid headline.

## 3. Prepare hourly news features

```bash
python preprocess_data.py btcusd_bitstamp_1min_2012-2025.csv.gz --news-file news_sentiment.csv --output btc_with_news.npz
```

The preprocessing script calculates the average sentiment score and number of articles for each hour. Hours without news receive sentiment 0 and count 0. These features are scaled using training-period statistics.

## 4. Train and compare

```bash
python forecast_btc.py --data btc_with_news.npz --model models/btc_with_news.keras --plot-dir results/with_news
```

Run the market-only experiment on the same market dates and compare the held-out test MAE and RMSE. The news model should be considered an improvement only if its test performance is better under the same split and evaluation procedure.

## Avoiding data leakage

- Use only articles that were available before the forecast target time.
- Preserve publication timestamps; do not use an article's crawl time if it was published later.
- Do not use future market movements to assign historical sentiment labels.
- Document the source, date range, filtering, duplicate handling, and sentiment method.
- Check the licence and terms of the news provider before collecting or redistributing headlines.

This repository provides a reproducible baseline for scoring and joining headlines. It does not automatically collect news, and it does not guarantee that news sentiment improves Bitcoin forecasts.
