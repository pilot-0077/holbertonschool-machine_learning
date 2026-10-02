#!/usr/bin/env python3
"""Train and validate an RNN that forecasts Bitcoin close prices."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


LOOKBACK_HOURS = 24
DATA_FILE = "btc_hourly.npz"
MODEL_FILE = "btc_forecaster.keras"


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


def main():
    """Train the model and report validation performance."""
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
    train_end = int(data["train_end"])

    if train_end <= LOOKBACK_HOURS:
        raise ValueError("Training data is too short.")
    if train_end >= len(features):
        raise ValueError("Validation data is empty.")

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

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            arguments.model,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
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

    print("Validation MSE: {:.6f}".format(metrics["loss"]))
    print("Validation MAE: {:.6f}".format(metrics["mae"]))

    target_mean = float(data["target_mean"])
    target_std = float(data["target_std"])
    normalized_rmse = np.sqrt(metrics["loss"])
    print(
        "Approx. validation RMSE in USD: ${:,.2f}".format(
            normalized_rmse * target_std
        )
    )
    print("Target mean used for scaling: ${:,.2f}".format(target_mean))


if __name__ == "__main__":
    main()
