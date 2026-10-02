#!/usr/bin/env python3
"""Preprocess raw Bitcoin minute data for time-series forecasting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]
TARGET = "Close"
TRAIN_FRACTION = 0.8
OUTPUT_FILE = "btc_hourly.npz"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Coinbase/Bitstamp minute data for forecasting."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Raw Coinbase and/or Bitstamp CSV files.",
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


def load_exchange(path):
    """Load one raw exchange CSV and validate required columns."""
    columns = ["Timestamp"] + FEATURES
    data = pd.read_csv(path, usecols=columns)

    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(
            "{} is missing columns: {}".format(
                path,
                ", ".join(sorted(missing)),
            )
        )

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        unit="s",
        errors="coerce",
    )
    data = data.dropna(subset=["Timestamp"])
    data = data.set_index("Timestamp").sort_index()
    return data


def to_hourly(data):
    """Aggregate one-minute OHLCV observations into hourly rows."""
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
    return hourly[FEATURES]


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

    weighted = np.divide(
        usd_volume,
        btc_volume,
        out=close.copy(),
        where=btc_volume != 0,
    )
    combined["Weighted_Price"] = weighted
    return combined[FEATURES]


def scale_data(data):
    """Scale features with statistics computed from training data only."""
    values = data.to_numpy(dtype=np.float64)
    train_end = int(len(values) * TRAIN_FRACTION)

    if train_end <= 24 or train_end >= len(values):
        raise ValueError("The dataset is too small for a chronological split.")

    train = values[:train_end]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0

    features = (values - mean) / std
    target_index = FEATURES.index(TARGET)
    target = features[:, target_index].copy()

    return features, target, mean, std, train_end


def main():
    """Run preprocessing and save the resulting arrays."""
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
    hourly = hourly.replace([np.inf, -np.inf], np.nan).dropna()

    features, target, mean, std, train_end = scale_data(hourly)
    timestamps = hourly.index.astype("int64").to_numpy() // 10**9
    target_index = FEATURES.index(TARGET)

    output = Path(arguments.output)
    np.savez_compressed(
        output,
        features=features.astype(np.float32),
        target=target.astype(np.float32),
        timestamps=timestamps.astype(np.int64),
        train_end=np.array(train_end, dtype=np.int64),
        feature_mean=mean,
        feature_std=std,
        target_mean=np.array(mean[target_index]),
        target_std=np.array(std[target_index]),
        feature_names=np.array(FEATURES),
    )

    print(
        "Saved {} hourly observations to {}.".format(
            len(features),
            output,
        )
    )
    print("Training rows: {}".format(train_end))
    print("Validation rows: {}".format(len(features) - train_end))


if __name__ == "__main__":
    main()
