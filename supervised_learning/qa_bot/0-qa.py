#!/usr/bin/env python3
"""Extractive question answering with a TensorFlow Hub BERT model."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


QA_MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
TOKENIZER_NAME = (
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
MAX_SEQ_LENGTH = 512
MAX_QUESTION_LENGTH = 64
MAX_ANSWER_LENGTH = 64
DOC_STRIDE = 128
MIN_ANSWER_CONFIDENCE = 1e-4

# Load the pretrained QA model and its matching WordPiece tokenizer once.
qa_model = hub.load(QA_MODEL_URL)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
qa_signature = qa_model.signatures["serving_default"]


def _run_qa_model(input_ids, input_mask, input_type_ids):
    """Run the Hub signature and return start and end logits."""
    signature_inputs = qa_signature.structured_input_signature[1]
    model_inputs = {}

    for name in signature_inputs:
        key = name.lower()
        if "word_ids" in key or "input_ids" in key:
            model_inputs[name] = input_ids
        elif "input_mask" in key or "attention_mask" in key:
            model_inputs[name] = input_mask
        elif "type_ids" in key or "segment_ids" in key:
            model_inputs[name] = input_type_ids

    # The model signature normally exposes the three standard BERT inputs.
    # Keep a positional fallback for signatures with non-standard names.
    if len(model_inputs) != len(signature_inputs):
        model_inputs = {
            name: value
            for name, value in zip(
                signature_inputs,
                (input_ids, input_mask, input_type_ids)
            )
        }

    outputs = qa_signature(**model_inputs)

    if isinstance(outputs, dict):
        output_names = list(outputs.keys())
        start_name = next(
            (name for name in output_names if "start" in name.lower()),
            None
        )
        end_name = next(
            (name for name in output_names if "end" in name.lower()),
            None
        )

        if start_name is not None and end_name is not None:
            start_logits = outputs[start_name]
            end_logits = outputs[end_name]
        else:
            start_logits, end_logits = list(outputs.values())[:2]
    else:
        start_logits, end_logits = outputs[:2]

    return (
        tf.reshape(start_logits, [-1]),
        tf.reshape(end_logits, [-1])
    )


def question_answer(question, reference):
    """Find and return an answer span from reference, or None."""
    if not isinstance(question, str) or not isinstance(reference, str):
        return None

    if not question.strip() or not reference.strip():
        return None

    question_tokens = tokenizer.tokenize(question)
    context_tokens = tokenizer.tokenize(reference)

    if not question_tokens or not context_tokens:
        return None

    question_tokens = question_tokens[:MAX_QUESTION_LENGTH]
    max_context_length = (
        MAX_SEQ_LENGTH - len(question_tokens) - 3
    )

    if max_context_length <= 0:
        return None

    best_answer = None
    best_confidence = 0.0
    context_start = len(question_tokens) + 2

    # Process the reference in overlapping windows so long documents are
    # not silently truncated at the model's 512-token input limit.
    chunk_start = 0
    while chunk_start < len(context_tokens):
        chunk_end = min(
            chunk_start + max_context_length,
            len(context_tokens)
        )
        context_chunk = context_tokens[chunk_start:chunk_end]

        tokens = (
            ["[CLS]"] + question_tokens + ["[SEP]"] +
            context_chunk + ["[SEP]"]
        )
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        type_ids = (
            [0] * (len(question_tokens) + 2) +
            [1] * (len(context_chunk) + 1)
        )
        attention_mask = [1] * len(token_ids)

        # Pad every input to the fixed BERT sequence length.
        padding_length = MAX_SEQ_LENGTH - len(token_ids)
        token_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        type_ids += [0] * padding_length

        input_ids = tf.constant([token_ids], dtype=tf.int32)
        input_mask = tf.constant([attention_mask], dtype=tf.int32)
        input_type_ids = tf.constant([type_ids], dtype=tf.int32)

        start_logits, end_logits = _run_qa_model(
            input_ids,
            input_mask,
            input_type_ids
        )

        start_probabilities = tf.nn.softmax(start_logits).numpy()
        end_probabilities = tf.nn.softmax(end_logits).numpy()

        context_end = context_start + len(context_chunk)
        # Search only positions belonging to the reference, never the
        # question, special tokens, or padding.
        for start_index in range(context_start, context_end):
            end_limit = min(
                context_end,
                start_index + MAX_ANSWER_LENGTH + 1
            )
            for end_index in range(start_index, end_limit):
                confidence = (
                    float(start_probabilities[start_index]) *
                    float(end_probabilities[end_index])
                )

                if confidence > best_confidence:
                    local_start = start_index - context_start
                    local_end = end_index - context_start
                    answer_tokens = context_chunk[
                        local_start:local_end + 1
                    ]
                    best_answer = tokenizer.convert_tokens_to_string(
                        answer_tokens
                    )
                    best_confidence = confidence

        if chunk_end >= len(context_tokens):
            break

        # Keep an overlap between windows to avoid splitting an answer.
        chunk_start = chunk_end - min(DOC_STRIDE, max_context_length // 2)

    if best_answer is None or best_confidence < MIN_ANSWER_CONFIDENCE:
        return None

    return best_answer.strip() or None
