#!/usr/bin/env python3
"""Create timestamped sentiment scores from Bitcoin news headlines."""

import argparse
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TIMESTAMP_ALIASES = {
    "timestamp",
    "published_at",
    "publishedat",
    "date",
    "time",
}
HEADLINE_ALIASES = {
    "headline",
    "title",
    "article_title",
    "text",
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score timestamped news headlines with VADER sentiment."
    )
    parser.add_argument(
        "input_csv",
        help="CSV containing a publication timestamp and a headline/title.",
    )
    parser.add_argument(
        "--output",
        default="news_sentiment.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def find_column(columns, aliases, label):
    """Find a column using case-insensitive aliases."""
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    raise ValueError(
        "Could not find a {} column. Expected one of: {}".format(
            label,
            ", ".join(sorted(aliases)),
        )
    )


def parse_timestamps(values):
    """Parse numeric Unix seconds or textual publication timestamps."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(
            values,
            unit="s",
            errors="coerce",
            utc=True,
        )
    return pd.to_datetime(values, errors="coerce", utc=True)


def main():
    """Read headlines and save one compound sentiment score per article."""
    arguments = parse_arguments()
    input_path = Path(arguments.input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(str(input_path))

    data = pd.read_csv(input_path)
    timestamp_column = find_column(
        data.columns,
        TIMESTAMP_ALIASES,
        "publication timestamp",
    )
    headline_column = find_column(
        data.columns,
        HEADLINE_ALIASES,
        "headline/title/text",
    )

    result = pd.DataFrame(
        {
            "Timestamp": parse_timestamps(data[timestamp_column]),
            "headline": data[headline_column].fillna("").astype(str),
        }
    )
    result["headline"] = result["headline"].str.strip()
    result = result.dropna(subset=["Timestamp"])
    result = result[result["headline"] != ""]

    if result.empty:
        raise ValueError(
            "No valid rows remain after parsing timestamps and headlines."
        )

    analyzer = SentimentIntensityAnalyzer()
    result["sentiment_score"] = result["headline"].map(
        lambda headline: analyzer.polarity_scores(headline)["compound"]
    )

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result[["Timestamp", "sentiment_score"]].to_csv(
        output_path,
        index=False,
    )

    print("Scored {} headlines.".format(len(result)))
    print("Sentiment CSV saved to: {}".format(output_path))
    print(
        "Note: VADER is a general-purpose baseline, not a finance-specific "
        "sentiment model."
    )


if __name__ == "__main__":
    main()
