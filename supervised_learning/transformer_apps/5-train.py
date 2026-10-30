#!/usr/bin/env python3
"""Custom training loop for the Portuguese-to-English Transformer."""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Transformer learning-rate schedule with linear warmup and decay."""

    def __init__(self, dm, warmup_steps=4000):
        """Store model dimensionality and warmup duration."""
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        """Calculate dm^(-0.5) * min(step^(-0.5), step*warmup^(-1.5))."""
        step = tf.cast(step, tf.float32)
        safe_step = tf.maximum(step, 1.0)
        arg1 = tf.math.rsqrt(safe_step)
        arg2 = step * tf.pow(self.warmup_steps, -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        """Return the schedule configuration."""
        return {
            'dm': self.dm,
            'warmup_steps': self.warmup_steps
        }


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a Transformer for Portuguese-to-English translation.

    Args:
        N: Number of encoder and decoder blocks.
        dm: Dimensionality of the model.
        h: Number of attention heads.
        hidden: Number of units in the feed-forward layers.
        max_len: Maximum token sequence length.
        batch_size: Number of examples per training batch.
        epochs: Number of training epochs.

    Returns:
        The trained Transformer model.
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden, input_vocab, target_vocab, max_len, max_len
    )

    learning_rate = CustomSchedule(dm, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inputs, target):
        """Run one teacher-forced training step and update model weights."""
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        encoder_mask, combined_mask, decoder_mask = create_masks(
            inputs, target_input
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inputs,
                target_input,
                True,
                encoder_mask,
                combined_mask,
                decoder_mask
            )
            token_losses = loss_object(target_real, predictions)
            token_mask = tf.cast(
                tf.not_equal(target_real, 0), token_losses.dtype
            )
            loss = tf.math.divide_no_nan(
                tf.reduce_sum(token_losses * token_mask),
                tf.reduce_sum(token_mask)
            )

        gradients = tape.gradient(
            loss, transformer.trainable_variables
        )
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        predicted_tokens = tf.argmax(
            predictions, axis=-1, output_type=target_real.dtype
        )
        correct_tokens = tf.cast(
            tf.equal(predicted_tokens, target_real), tf.float32
        )
        accuracy_mask = tf.cast(token_mask, tf.float32)
        accuracy = tf.math.divide_no_nan(
            tf.reduce_sum(correct_tokens * accuracy_mask),
            tf.reduce_sum(accuracy_mask)
        )

        train_loss.update_state(loss)
        train_accuracy.update_state(accuracy)

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch_number, (inputs, target) in enumerate(data.data_train):
            train_step(inputs, target)

            if batch_number % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, batch {batch_number}: "
                    f"loss {float(train_loss.result().numpy())} "
                    f"accuracy {float(train_accuracy.result().numpy())}"
                )

        print(
            f"Epoch {epoch + 1}: "
            f"loss {float(train_loss.result().numpy())} "
            f"accuracy {float(train_accuracy.result().numpy())}"
        )

    return transformer
