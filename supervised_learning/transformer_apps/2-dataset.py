#!/usr/bin/env python3
"""Load, tokenize, and TensorFlow-encode the translation dataset."""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Prepare Portuguese-English data for Transformer applications."""

    def __init__(self):
        """Load splits, train tokenizers, and encode both datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Train Portuguese and English tokenizers from the training data.

        Args:
            data: Dataset of pairs of Portuguese and English string tensors.

        Returns:
            The Portuguese tokenizer and the English tokenizer.
        """
        pt_sentences = []
        en_sentences = []

        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese-English pair with start/end token IDs.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing the corresponding English sentence.

        Returns:
            Two integer lists: Portuguese tokens and English tokens.
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wrap encode() in TensorFlow and restore dynamic vector shapes.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing the corresponding English sentence.

        Returns:
            Portuguese and English token ID tensors, each with shape (None,).
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
