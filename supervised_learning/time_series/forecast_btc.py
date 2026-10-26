#!/usr/bin/env python3
"""Train, evaluate, and plot a GRU model for Bitcoin price forecasting."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
        help="Directory in which evaluation plots will be saved.",
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
            "target_mean",
            "target_std",
            "timestamps",
            "feature_names",
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
    plt.ylabel("Mean squared error (normalized target)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_prediction_plot(timestamps, actual, predicted, baseline, output_path):
    """Save actual, model-predicted, and baseline prices for validation."""
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
    plt.title("Actual vs Predicted Bitcoin Closing Prices")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    """Train the model, report metrics, and save evaluation plots."""
    arguments = parse_arguments()

    if arguments.epochs < 1:
        raise ValueError("epochs must be positive.")
    if arguments.batch_size < 1:
        raise ValueError("batch-size must be positive.")

    tf.keras.utils.set_random_seed(0)

    data_path = Path(arguments.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            "{} not found. Run preprocess_data.py first.".format(
                data_path
            )
        )

    data = load_data(data_path)
    features = data["features"].astype(np.float32)
    target = data["target"].astype(np.float32)
    timestamps = np.asarray(data["timestamps"]).reshape(-1)
    feature_names = [str(name) for name in data["feature_names"]]
    train_end = int(data["train_end"])

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if len(target) != len(features) or len(timestamps) != len(features):
        raise ValueError(
            "features, target, and timestamps must have equal lengths."
        )
    if "Close" not in feature_names:
        raise ValueError("The feature list must contain a Close column.")
    if train_end <= LOOKBACK_HOURS:
        raise ValueError("Training data is too short.")
    if train_end >= len(features):
        raise ValueError("Validation data is empty.")

    close_index = feature_names.index("Close")
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

    metrics = model.evaluate(
        validation_dataset,
        return_dict=True,
        verbose=0,
    )
    predictions_scaled = model.predict(
        validation_dataset,
        verbose=0,
    ).reshape(-1)
    actual_scaled = target[train_end:]

    if len(predictions_scaled) != len(actual_scaled):
        raise RuntimeError(
            "Prediction count does not match the validation target count."
        )

    target_mean = float(data["target_mean"])
    target_std = float(data["target_std"])
    actual_usd = actual_scaled * target_std + target_mean
    predicted_usd = predictions_scaled * target_std + target_mean

    # A simple baseline predicts that the next close equals the latest close.
    baseline_scaled = features[
        train_end - 1:len(features) - 1,
        close_index,
    ]
    baseline_usd = baseline_scaled * target_std + target_mean

    model_mae_usd = float(np.mean(np.abs(predicted_usd - actual_usd)))
    model_rmse_usd = float(np.sqrt(np.mean((predicted_usd - actual_usd) ** 2)))
    baseline_mae_usd = float(np.mean(np.abs(baseline_usd - actual_usd)))
    baseline_rmse_usd = float(
        np.sqrt(np.mean((baseline_usd - actual_usd) ** 2))
    )

    loss_plot = plot_dir / "training_validation_loss.png"
    prediction_plot = plot_dir / "actual_vs_predicted.png"
    save_loss_plot(history, loss_plot)
    save_prediction_plot(
        timestamps[train_end:],
        actual_usd,
        predicted_usd,
        baseline_usd,
        prediction_plot,
    )

    print("Validation MSE (normalized): {:.6f}".format(metrics["loss"]))
    print("Validation MAE (normalized): {:.6f}".format(metrics["mae"]))
    print("Validation MAE (USD): ${:,.2f}".format(model_mae_usd))
    print("Validation RMSE (USD): ${:,.2f}".format(model_rmse_usd))
    print("Naive baseline MAE (USD): ${:,.2f}".format(baseline_mae_usd))
    print("Naive baseline RMSE (USD): ${:,.2f}".format(baseline_rmse_usd))
    print("Best model saved to: {}".format(model_path))
    print("Loss plot saved to: {}".format(loss_plot))
    print("Prediction plot saved to: {}".format(prediction_plot))


if __name__ == "__main__":
    main()
