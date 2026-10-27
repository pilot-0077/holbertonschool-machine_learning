#!/usr/bin/env python3
"""Prepare Bitcoin market data for time-series forecasting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]
NEWS_FEATURES = ["News_Sentiment", "News_Count"]
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
OUTPUT_FILE = "btc_hourly.npz"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Bitcoin market and optional news data."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Raw Coinbase and/or Bitstamp CSV files.",
    )
    parser.add_argument(
        "--news-file",
        default=None,
        help=(
            "Optional CSV with Timestamp and sentiment_score columns, "
            "one row per news article."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive start timestamp in UTC.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end timestamp in UTC.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Path for the compressed preprocessed dataset.",
    )
    return parser.parse_args()


def discover_csv_files():
    """Find common Bitcoin one-minute CSV files in the current directory."""
    patterns = (
        "*coinbase*1-min*.csv",
        "*bitstamp*1-min*.csv",
        "*Coinbase*1-min*.csv",
        "*Bitstamp*1-min*.csv",
    )
    found = []
    for pattern in patterns:
        found.extend(Path(".").glob(pattern))
    return sorted(set(found))


def parse_boundary(value):
    """Parse an optional UTC timestamp boundary."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def load_exchange(path):
    """Load exchange CSVs using either Kaggle or simple OHLCV headers."""
    data = pd.read_csv(path)
    aliases = {
        "Timestamp": {"timestamp", "time"},
        "Open": {"open"},
        "High": {"high"},
        "Low": {"low"},
        "Close": {"close"},
        "Volume_(BTC)": {"volume_(btc)", "volume", "volume_btc"},
        "Volume_(Currency)": {
            "volume_(currency)",
            "volume_currency",
            "volume_usd",
            "quote_volume",
        },
    }

    rename = {}
    lowered = {
        str(column).strip().lower(): column
        for column in data.columns
    }
    for canonical, names in aliases.items():
        for alias in names:
            if alias in lowered:
                rename[lowered[alias]] = canonical
                break
    data = data.rename(columns=rename)

    required = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume_(BTC)",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "{} is missing required columns: {}".format(
                path,
                ", ".join(sorted(missing)),
            )
        )

    if pd.api.types.is_numeric_dtype(data["Timestamp"]):
        data["Timestamp"] = pd.to_datetime(
            data["Timestamp"],
            unit="s",
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
    else:
        data["Timestamp"] = pd.to_datetime(
            data["Timestamp"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume_(BTC)",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "Volume_(Currency)" not in data.columns:
        data["Volume_(Currency)"] = (
            data["Volume_(BTC)"] * data["Close"]
        )
    else:
        data["Volume_(Currency)"] = pd.to_numeric(
            data["Volume_(Currency)"],
            errors="coerce",
        )

    data = data.dropna(subset=["Timestamp", "Close"])
    data = data.set_index("Timestamp").sort_index()
    return data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume_(BTC)",
            "Volume_(Currency)",
        ]
    ]


def to_hourly(data):
    """Aggregate one-minute observations into hourly market rows."""
    hourly = data.resample("1h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
        }
    )

    first_valid = hourly["Close"].first_valid_index()
    if first_valid is None:
        raise ValueError("The dataset contains no valid close prices.")

    hourly = hourly.loc[first_valid:]
    hourly["Close"] = hourly["Close"].ffill()

    for column in ("Open", "High", "Low"):
        hourly[column] = hourly[column].fillna(hourly["Close"])

    for column in ("Volume_(BTC)", "Volume_(Currency)"):
        hourly[column] = hourly[column].fillna(0.0)

    btc_volume = hourly["Volume_(BTC)"].to_numpy()
    usd_volume = hourly["Volume_(Currency)"].to_numpy()
    close = hourly["Close"].to_numpy()

    weighted = np.divide(
        usd_volume,
        btc_volume,
        out=close.copy(),
        where=btc_volume != 0,
    )
    hourly["Weighted_Price"] = weighted
    return hourly[MARKET_FEATURES]


def combine_exchanges(frames):
    """Combine hourly observations from one or more exchanges."""
    if len(frames) == 1:
        return frames[0]

    data = pd.concat(frames).sort_index()
    grouped = data.groupby(level=0)
    combined = grouped.agg(
        {
            "Open": "mean",
            "High": "max",
            "Low": "min",
            "Close": "mean",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
        }
    )

    btc_volume = combined["Volume_(BTC)"].to_numpy()
    usd_volume = combined["Volume_(Currency)"].to_numpy()
    close = combined["Close"].to_numpy()

    combined["Weighted_Price"] = np.divide(
        usd_volume,
        btc_volume,
        out=close.copy(),
        where=btc_volume != 0,
    )
    return combined[MARKET_FEATURES]


def load_hourly_news(path, hourly_index):
    """Align timestamped article sentiment scores to hourly market rows."""
    news = pd.read_csv(path)
    timestamp_aliases = {
        "timestamp",
        "date_time",
        "published_at",
        "publishedat",
        "date",
        "time",
    }
    score_aliases = {"sentiment_score", "score", "sentiment"}
    label_aliases = {"sentiment_label", "label", "sentiment_class", "class"}
    # Normalize spaces and hyphens so headers such as "Date Time" and
    # "Coin Type" match the canonical aliases below.
    column_lookup = {
        str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
        for column in news.columns
    }
    timestamp_column = next(
        (
            column_lookup[name]
            for name in timestamp_aliases
            if name in column_lookup
        ),
        None,
    )
    score_column = next(
        (
            column_lookup[name]
            for name in score_aliases
            if name in column_lookup
        ),
        None,
    )
    label_column = next(
        (
            column_lookup[name]
            for name in label_aliases
            if name in column_lookup
        ),
        None,
    )
    if timestamp_column is None or score_column is None:
        raise ValueError(
            "{} must contain a timestamp column and a sentiment score "
            "column.".format(path)
        )

    news = news.rename(
        columns={
            timestamp_column: "Timestamp",
            score_column: "sentiment_score",
        }
    )

    coin_aliases = {"coin_type", "coin", "asset", "symbol"}
    coin_column = next(
        (
            column_lookup[name]
            for name in coin_aliases
            if name in column_lookup
        ),
        None,
    )
    if coin_column is not None:
        is_bitcoin = news[coin_column].astype(str).str.lower().str.contains(
            r"\bbitcoin\b|\bbtc\b",
            regex=True,
        )
        news = news.loc[is_bitcoin]
        if news.empty:
            raise ValueError(
                "No Bitcoin-specific rows were found in {}.".format(path)
            )

    if pd.api.types.is_numeric_dtype(news["Timestamp"]):
        news["Timestamp"] = pd.to_datetime(
            news["Timestamp"],
            unit="s",
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
    else:
        news["Timestamp"] = pd.to_datetime(
            news["Timestamp"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

    raw_scores = pd.to_numeric(
        news["sentiment_score"],
        errors="coerce",
    )
    if label_column is not None:
        # CryptoVision's score is confidence, while the label supplies the
        # direction. Convert confidence into signed sentiment: positive > 0,
        # negative < 0, and neutral = 0.
        label_direction = news[label_column].astype(str).str.strip().str.lower().map(
            {
                "positive": 1.0,
                "pos": 1.0,
                "bullish": 1.0,
                "negative": -1.0,
                "neg": -1.0,
                "bearish": -1.0,
                "neutral": 0.0,
                "neu": 0.0,
                "stable": 0.0,
            }
        )
        confidence = raw_scores.abs().clip(lower=0.0, upper=1.0)
        news["sentiment_score"] = label_direction * confidence
    else:
        # Score-only files (for example, VADER output) may already contain
        # signed sentiment values.
        news["sentiment_score"] = raw_scores

    news = news.dropna(subset=["Timestamp", "sentiment_score"])

    if news.empty:
        raise ValueError(
            "No valid timestamped news sentiment rows were found."
        )

    if (
        news["Timestamp"].max() < hourly_index.min()
        or news["Timestamp"].min() > hourly_index.max()
    ):
        raise ValueError(
            "News timestamps do not overlap the selected market date range."
        )

    news = news.set_index("Timestamp").sort_index()
    hourly_news = news.resample("1h").agg(
        News_Sentiment=("sentiment_score", "mean"),
        News_Count=("sentiment_score", "count"),
    )
    hourly_news = hourly_news.reindex(hourly_index)
    hourly_news["News_Sentiment"] = (
        hourly_news["News_Sentiment"].fillna(0.0)
    )
    hourly_news["News_Count"] = hourly_news["News_Count"].fillna(0.0)
    return hourly_news[NEWS_FEATURES]


def scale_data(data, target_values):
    """Scale features and next-hour log returns using training-only stats."""
    values = data.to_numpy(dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    train_end = int(len(values) * TRAIN_FRACTION)
    validation_end = int(
        len(values) * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    if (
        train_end <= 24
        or validation_end <= train_end
        or validation_end >= len(values)
    ):
        raise ValueError(
            "The dataset is too small for train/validation/test splits."
        )

    train_features = values[:train_end]
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    features = (values - feature_mean) / feature_std

    train_targets = target_values[:train_end]
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())
    if target_std == 0:
        target_std = 1.0
    target = (target_values - target_mean) / target_std

    return (
        features,
        target,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        train_end,
        validation_end,
    )


def main():
    """Prepare scale-independent market features and next-hour log returns."""
    arguments = parse_arguments()
    csv_files = [Path(path) for path in arguments.csv_files]

    if not csv_files:
        csv_files = discover_csv_files()
    if not csv_files:
        raise FileNotFoundError(
            "No Coinbase or Bitstamp one-minute CSV file was found."
        )

    frames = []
    for path in csv_files:
        if not path.is_file():
            raise FileNotFoundError(str(path))
        print("Loading {}...".format(path))
        frames.append(to_hourly(load_exchange(path)))

    hourly = combine_exchanges(frames)
    start_date = parse_boundary(arguments.start_date)
    end_date = parse_boundary(arguments.end_date)
    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise ValueError("--start-date must not be after --end-date.")
    if start_date is not None:
        hourly = hourly.loc[hourly.index >= start_date]
    if end_date is not None:
        hourly = hourly.loc[hourly.index <= end_date]
    if hourly.empty:
        raise ValueError("No market observations remain after date filtering.")

    close = hourly["Close"].astype(float).replace(0.0, np.nan)
    log_return = np.log(close).diff()
    feature_data = pd.DataFrame(index=hourly.index)
    feature_data["Open_vs_Close"] = hourly["Open"] / close - 1.0
    feature_data["High_vs_Close"] = hourly["High"] / close - 1.0
    feature_data["Low_vs_Close"] = hourly["Low"] / close - 1.0
    feature_data["Close_Return_1h"] = log_return
    feature_data["Volume_BTC_Log"] = np.log1p(
        hourly["Volume_(BTC)"].clip(lower=0.0)
    )
    feature_data["Volume_Currency_Log"] = np.log1p(
        hourly["Volume_(Currency)"].clip(lower=0.0)
    )
    feature_data["Weighted_Price_vs_Close"] = (
        hourly["Weighted_Price"] / close - 1.0
    )

    if arguments.news_file:
        news_path = Path(arguments.news_file)
        if not news_path.is_file():
            raise FileNotFoundError(str(news_path))
        news = load_hourly_news(news_path, hourly.index)
        feature_data = feature_data.join(news)
        print(
            "Added hourly news sentiment features from {}.".format(
                news_path
            )
        )

    aligned = feature_data.join(log_return.rename("target_return"))
    aligned = aligned.join(close.rename("close_usd"))
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        raise ValueError("No valid feature and target rows remain.")

    feature_names = list(feature_data.columns)
    target_values = aligned["target_return"].to_numpy(dtype=np.float64)
    (
        features,
        target,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        train_end,
        validation_end,
    ) = scale_data(aligned[feature_names], target_values)

    timestamps = aligned.index.astype("int64").to_numpy() // 10**9
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        features=features.astype(np.float32),
        target=target.astype(np.float32),
        close_usd=aligned["close_usd"].to_numpy(dtype=np.float64),
        timestamps=timestamps.astype(np.int64),
        train_end=np.array(train_end, dtype=np.int64),
        validation_end=np.array(validation_end, dtype=np.int64),
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=np.array(target_mean),
        target_std=np.array(target_std),
        feature_names=np.array(feature_names),
    )

    total = len(features)
    print("Saved {} hourly observations to {}.".format(total, output))
    print("Target: next-hour log return, converted to a closing price.")
    print("Training rows: {} (70%)".format(train_end))
    print(
        "Validation rows: {} (15%)".format(
            validation_end - train_end
        )
    )
    print("Test rows: {} (15%)".format(total - validation_end))


if __name__ == "__main__":
    main()