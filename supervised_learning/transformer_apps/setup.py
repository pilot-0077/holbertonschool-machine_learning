#!/usr/bin/env python3
"""Load the manually downloaded TED Portuguese-English translation dataset.

This helper replaces the unavailable TFDS download for
ted_hrlr_translate/pt_to_en and yields the same pairs of tf.string tensors.
"""
import os
import sys
from pathlib import Path

import tensorflow as tf

SOURCE_LANG = "pt"
TARGET_LANG = "en"
PAIR_DIR = "pt_to_en"

_SPLIT_SUFFIX = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


def _candidate_roots():
    """Return candidate roots where the extracted archive may be located."""
    roots = []
    env = os.environ.get("TED_HRLR_DIR")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "ted_hrlr")
    roots.append(Path.cwd())
    return roots


def resolve_data_dir():
    """Find the extracted datasets/pt_to_en directory."""
    for root in _candidate_roots():
        data_dir = root / "datasets" / PAIR_DIR
        if data_dir.is_dir() and any(data_dir.glob(f"{SOURCE_LANG}.*")):
            return data_dir

    looked = "\n  ".join(
        str(root / "datasets" / PAIR_DIR) for root in _candidate_roots()
    )
    sys.exit(
        "Extracted TED translation data not found. Looked in:\n  "
        + looked
        + "\n\nExtract the archive into ~/.cache/ted_hrlr with:\n"
        "  mkdir -p ~/.cache/ted_hrlr\n"
        "  tar -xzvf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr"
    )


def _read_pairs(data_dir, split):
    """Yield non-empty Portuguese/English sentence pairs for a split."""
    suffix = _SPLIT_SUFFIX[split]
    src_file = data_dir / f"{SOURCE_LANG}.{suffix}"
    tgt_file = data_dir / f"{TARGET_LANG}.{suffix}"

    for file_path in (src_file, tgt_file):
        if not file_path.is_file():
            sys.exit(f"Missing TED translation data file: {file_path}")

    src_sentences = src_file.read_text(encoding="utf-8").split("\n")
    tgt_sentences = tgt_file.read_text(encoding="utf-8").split("\n")

    if len(src_sentences) != len(tgt_sentences):
        sys.exit(
            f"Line count mismatch for {split}: "
            f"{len(src_sentences)} ({src_file.name}) vs "
            f"{len(tgt_sentences)} ({tgt_file.name})"
        )

    for pt, en in zip(src_sentences, tgt_sentences):
        if pt and en:
            yield pt, en


def load_pt2en(split="train"):
    """Return a tf.data.Dataset of (Portuguese, English) tf.string pairs."""
    if split not in _SPLIT_SUFFIX:
        raise ValueError(
            f"split must be one of {list(_SPLIT_SUFFIX)}, got {split!r}"
        )

    data_dir = resolve_data_dir()

    def generator():
        """Generate sentence pairs for TensorFlow."""
        yield from _read_pairs(data_dir, split)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )


if __name__ == "__main__":
    data_directory = resolve_data_dir()
    print(f"Using data dir: {data_directory}")
    for split in _SPLIT_SUFFIX:
        count = sum(1 for _ in _read_pairs(data_directory, split))
        print(f"{split}: {count} pairs")
    print("First training example:")
    for pt, en in load_pt2en("train").take(1):
        print("PT:", pt.numpy().decode("utf-8"))
        print("EN:", en.numpy().decode("utf-8"))
