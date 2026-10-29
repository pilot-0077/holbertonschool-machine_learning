#!/usr/bin/env python3
"""Load and prepare a dataset for machine translation."""
import transformers
from setup import load_pt2en


class Dataset:
    """Load the Portuguese-to-English TED Talks dataset and tokenizers."""

    def __init__(self):
        """Initialize training/validation data and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Train Portuguese and English subword tokenizers from data.

        Args:
            data: A tf.data.Dataset of (Portuguese, English) string tensors.

        Returns:
            A tuple containing the Portuguese and English tokenizers.
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
