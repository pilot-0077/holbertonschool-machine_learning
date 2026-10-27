#!/usr/bin/env python3
"""Train a GRU to forecast next-hour Bitcoin log returns."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


LOOKBACK_HOURS = 24
DATA_FILE = "btc_hourly.npz"
MODEL_FILE = "btc_forecaster.keras"
PLOT_DIR = "plots"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Forecast the next hourly Bitcoin close price."
    )
    parser.add_argument(
        "--data",
        default=DATA_FILE,
        help="Preprocessed NPZ file created by preprocess_data.py.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_FILE,
        help="Path used to save the best trained model.",
    )
    parser.add_argument(
        "--plot-dir",
        default=PLOT_DIR,
        help="Directory in which evaluation outputs will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    return parser.parse_args()


def load_data(path):
    """Load and validate the preprocessed time-series arrays."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features",
            "target",
            "train_end",
            "validation_end",
            "target_mean",
            "target_std",
            "timestamps",
            "feature_names",
            "close_usd",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "Missing arrays in {}: {}".format(
                    path,
                    ", ".join(sorted(missing)),
                )
            )
        return {key: archive[key] for key in archive.files}


def make_dataset(
    features,
    target,
    target_start,
    target_end,
    batch_size,
    shuffle,
):
    """Create 24-hour input windows and next-hour targets."""
    if target_start < LOOKBACK_HOURS:
        raise ValueError("target_start is too early for a 24-hour window.")
    if target_start >= target_end:
        raise ValueError("The requested dataset split is empty.")

    window_data = features[
        target_start - LOOKBACK_HOURS:target_end - 1
    ]
    labels = target[target_start:target_end]

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=window_data,
        targets=labels,
        sequence_length=LOOKBACK_HOURS,
        sequence_stride=1,
        sampling_rate=1,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(feature_count):
    """Create and compile a GRU-based forecasting model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(LOOKBACK_HOURS, feature_count)
            ),
            tf.keras.layers.GRU(
                64,
                return_sequences=True,
            ),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),
            tf.keras.layers.Dense(1),
        ],
        name="btc_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def save_loss_plot(history, output_path):
    """Save training and validation loss over the training epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error (normalized log return)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_prediction_plot(timestamps, actual, predicted, baseline, output_path):
    """Save actual, predicted, and baseline prices for the test period."""
    dates = pd.to_datetime(timestamps, unit="s", utc=True)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual close", linewidth=1.5)
    plt.plot(dates, predicted, label="GRU prediction", linewidth=1.2)
    plt.plot(
        dates,
        baseline,
        label="Naive baseline (previous close)",
        linewidth=1.0,
        alpha=0.8,
    )
    plt.xlabel("Time (UTC)")
    plt.ylabel("Bitcoin closing price (USD)")
    plt.title("Actual vs Predicted Bitcoin Closing Prices (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    """Train on the train split, select on validation, and evaluate on test."""
    arguments = parse_arguments()
    if arguments.epochs < 1:
        raise ValueError("epochs must be positive.")
    if arguments.batch_size < 1:
        raise ValueError("batch-size must be positive.")

    tf.keras.utils.set_random_seed(0)
    data_path = Path(arguments.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            "{} not found. Run preprocess_data.py first.".format(data_path)
        )

    data = load_data(data_path)
    features = data["features"].astype(np.float32)
    target = data["target"].astype(np.float32)
    close_usd = np.asarray(data["close_usd"], dtype=np.float64).reshape(-1)
    timestamps = np.asarray(data["timestamps"]).reshape(-1)
    train_end = int(data["train_end"])
    validation_end = int(data["validation_end"])

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if (
        len(target) != len(features)
        or len(timestamps) != len(features)
        or len(close_usd) != len(features)
    ):
        raise ValueError(
            "features, target, timestamps, and close_usd must have equal "
            "lengths."
        )
    if train_end <= LOOKBACK_HOURS:
        raise ValueError("Training data is too short.")
    if validation_end <= train_end or validation_end >= len(features):
        raise ValueError("Validation or test split is empty.")
    if np.any(close_usd <= 0):
        raise ValueError("All closing prices must be positive.")

    train_dataset = make_dataset(
        features,
        target,
        LOOKBACK_HOURS,
        train_end,
        arguments.batch_size,
        True,
    )
    validation_dataset = make_dataset(
        features,
        target,
        train_end,
        validation_end,
        arguments.batch_size,
        False,
    )
    test_dataset = make_dataset(
        features,
        target,
        validation_end,
        len(features),
        arguments.batch_size,
        False,
    )

    model = build_model(features.shape[1])
    model.summary()

    model_path = Path(arguments.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(arguments.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=arguments.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    validation_metrics = model.evaluate(
        validation_dataset,
        return_dict=True,
        verbose=0,
    )
    test_metrics = model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=0,
    )
    predicted_returns_scaled = model.predict(
        test_dataset,
        verbose=0,
    ).reshape(-1)
    actual_returns_scaled = target[validation_end:]

    if len(predicted_returns_scaled) != len(actual_returns_scaled):
        raise RuntimeError(
            "Prediction count does not match the test target count."
        )

    target_mean = float(data["target_mean"])
    target_std = float(data["target_std"])
    predicted_log_returns = (
        predicted_returns_scaled * target_std + target_mean
    )
    actual_log_returns = actual_returns_scaled * target_std + target_mean

    # Convert predicted next-hour log returns into closing-price estimates.
    previous_close_usd = close_usd[validation_end - 1:-1]
    actual_close_usd = close_usd[validation_end:]
    predicted_close_usd = previous_close_usd * np.exp(
        np.clip(predicted_log_returns, -20.0, 20.0)
    )
    # Naive baseline: next close equals the most recent observed close.
    baseline_close_usd = previous_close_usd.copy()

    model_mae_usd = float(
        np.mean(np.abs(predicted_close_usd - actual_close_usd))
    )
    model_rmse_usd = float(
        np.sqrt(np.mean((predicted_close_usd - actual_close_usd) ** 2))
    )
    baseline_mae_usd = float(
        np.mean(np.abs(baseline_close_usd - actual_close_usd))
    )
    baseline_rmse_usd = float(
        np.sqrt(np.mean((baseline_close_usd - actual_close_usd) ** 2))
    )

    loss_plot = plot_dir / "training_validation_loss.png"
    prediction_plot = plot_dir / "actual_vs_predicted_test.png"
    predictions_csv = plot_dir / "test_predictions.csv"
    save_loss_plot(history, loss_plot)
    save_prediction_plot(
        timestamps[validation_end:],
        actual_close_usd,
        predicted_close_usd,
        baseline_close_usd,
        prediction_plot,
    )
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                timestamps[validation_end:],
                unit="s",
                utc=True,
            ),
            "actual_close_usd": actual_close_usd,
            "gru_prediction_usd": predicted_close_usd,
            "naive_baseline_usd": baseline_close_usd,
            "actual_log_return": actual_log_returns,
            "predicted_log_return": predicted_log_returns,
        }
    ).to_csv(predictions_csv, index=False)

    print("Target: next-hour log return.")
    print("Validation MSE (normalized return): {:.6f}".format(
        validation_metrics["loss"]
    ))
    print("Validation MAE (normalized return): {:.6f}".format(
        validation_metrics["mae"]
    ))
    print("Test MSE (normalized return): {:.6f}".format(test_metrics["loss"]))
    print("Test MAE (normalized return): {:.6f}".format(test_metrics["mae"]))
    print("Test MAE (USD): {} USD".format(format(model_mae_usd, ",.2f")))
    print("Test RMSE (USD): {} USD".format(format(model_rmse_usd, ",.2f")))
    print("Naive baseline MAE (USD): {} USD".format(
        format(baseline_mae_usd, ",.2f")
    ))
    print("Naive baseline RMSE (USD): {} USD".format(
        format(baseline_rmse_usd, ",.2f")
    ))
    print("Best model saved to: {}".format(model_path))
    print("Loss plot saved to: {}".format(loss_plot))
    print("Test prediction plot saved to: {}".format(prediction_plot))
    print("Test predictions saved to: {}".format(predictions_csv))

if __name__ == "__main__":
    main()
