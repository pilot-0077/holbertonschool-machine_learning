#!/usr/bin/env python3
"""Neural style transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST object."""
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        self.style_image = self.scale_image(style_image)

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.beta = beta

        self.model = None
        self.load_model()
        self.gram_style_features, self.content_feature = (
            self.generate_features()
        )

    @staticmethod
    def scale_image(image):
        """Scale image values to [0, 1] with largest side equal to 512."""
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = int((h * 512) / w)
        else:
            h_new = 512
            w_new = int((w * 512) / h)

        resized_image = tf.image.resize(
            image,
            size=[h_new, w_new],
            method='bicubic'
        )
        resized_image = resized_image / 255.0
        resized_image = tf.clip_by_value(resized_image, 0, 1)

        return tf.expand_dims(resized_image, 0)

    def load_model(self):
        """Create the model used to calculate neural style costs."""
        model_vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        model_vgg19.trainable = False

        selected_layers = self.style_layers + [self.content_layer]
        outputs = [
            model_vgg19.get_layer(name).output
            for name in selected_layers
        ]
        model = tf.keras.Model([model_vgg19.input], outputs)

        custom_objects = {
            'MaxPooling2D': tf.keras.layers.AveragePooling2D
        }
        tf.keras.models.save_model(model, 'vgg_base.h5')
        self.model = tf.keras.models.load_model(
            'vgg_base.h5',
            custom_objects=custom_objects
        )

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of an input layer."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        reshaped = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(reshaped, reshaped, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)

        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """Extract the style and content features used for NST costs."""
        preprocess_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        preprocess_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_output = self.model(preprocess_style)
        content_output = self.model(preprocess_content)

        self.gram_style_features = [
            self.gram_matrix(style_layer)
            for style_layer in style_output
        ]
        self.gram_style_features = self.gram_style_features[:-1]
        self.content_feature = content_output[-1]

        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for a single style layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != [1, c, c]):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c
                )
            )

        output_gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(
            tf.square(output_gram_style - gram_target)
        )

    def style_cost(self, style_outputs):
        """Calculate the style cost for the generated image."""
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1.0 / float(length)
        costs = [
            weight * self.layer_style_cost(style, target)
            for style, target in zip(
                style_outputs,
                self.gram_style_features
            )
        ]

        return sum(costs)

    def content_cost(self, content_output):
        """Calculate the content cost for the generated image."""
        content_shape = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    content_shape
                )
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    def total_cost(self, generated_image):
        """Calculate total, content, and style costs."""
        content_shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != content_shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    content_shape
                )
            )

        preprocess_generated = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )
        generated_output = self.model(preprocess_generated)
        generated_content = generated_output[-1]
        generated_style = generated_output[:-1]

        j_content = self.content_cost(generated_content)
        j_style = self.style_cost(generated_style)
        j_total = self.alpha * j_content + self.beta * j_style

        return j_total, j_content, j_style

    def compute_grads(self, generated_image):
        """Calculate gradients for the generated image."""
        content_shape = (
            1,
            self.content_image.shape[1],
            self.content_image.shape[2],
            3
        )
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != content_shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    content_shape
                )
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            j_total, j_content, j_style = self.total_cost(generated_image)

        gradients = tape.gradient(j_total, generated_image)
        return gradients, j_total, j_content, j_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate the neural style transferred image using Adam."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError(
                "step must be positive and less than iterations"
            )

        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = generated_image.numpy().copy()

        for i in range(iterations + 1):
            gradients, j_total, j_content, j_style = self.compute_grads(
                generated_image
            )

            if float(j_total) < float(best_cost):
                best_cost = j_total
                best_image = generated_image.numpy().copy()

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}".format(
                        i, j_total, j_content, j_style
                    )
                )

            if i != iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0, 1)
                )

        if isinstance(best_cost, (tf.Tensor, tf.Variable)):
            best_cost = best_cost.numpy()

        return best_image[0], best_cost
