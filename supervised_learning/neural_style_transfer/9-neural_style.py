#!/usr/bin/env python3
"""Neural style transfer implementation."""

import numpy as np
import tensorflow as tf


class NST:
    """Perform neural style transfer using a VGG19 feature extractor."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the neural style transfer object."""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = None
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Scale an image to [0, 1] with its largest side equal to 512."""
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        height, width, _ = image.shape
        scale = 512 / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        image = tf.convert_to_tensor(image)
        image = tf.image.resize(
            image,
            [new_height, new_width],
            method='bicubic'
        )
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Load VGG19 and expose the required style/content outputs."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

        custom_objects = {
            'MaxPooling2D': tf.keras.layers.AveragePooling2D
        }
        tf.keras.models.save_model(model, 'vgg_base.h5')
        self.model = tf.keras.models.load_model(
            'vgg_base.h5',
            custom_objects=custom_objects
        )
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the normalized Gram matrix of a rank-4 tensor."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )
        shape = tf.shape(input_layer)
        locations = tf.cast(shape[1] * shape[2], tf.float32)
        return gram / locations

    def generate_features(self):
        """Extract target style Gram matrices and the content feature."""
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]
        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for one VGG style layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        channels = style_output.shape[-1]
        expected = (1, channels, channels)
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or tuple(gram_target.shape) != expected):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    channels, channels
                )
            )

        gram = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram - gram_target))

    def style_cost(self, style_outputs):
        """Calculate the evenly weighted style cost over all style layers."""
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1.0 / length
        costs = [
            weight * self.layer_style_cost(output, target)
            for output, target in zip(
                style_outputs,
                self.gram_style_features
            )
        ]
        return tf.add_n(costs)

    def content_cost(self, content_output):
        """Calculate the content cost for a generated content output."""
        expected = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != expected):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    expected
                )
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    def total_cost(self, generated_image):
        """Calculate total, content, and style costs."""
        expected = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != expected):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    expected
                )
            )

        image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )
        outputs = self.model(image)
        content = self.content_cost(outputs[-1])
        style = self.style_cost(outputs[:-1])
        total = self.alpha * content + self.beta * style
        return total, content, style

    def compute_grads(self, generated_image):
        """Calculate gradients of total cost with respect to the image."""
        expected = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != expected):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    expected
                )
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            total, content, style = self.total_cost(generated_image)

        gradients = tape.gradient(total, generated_image)
        return gradients, total, content, style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate a neural style transferred image with Adam."""
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
        best_image = generated_image.numpy()

        for iteration in range(iterations + 1):
            gradients, total, content, style = self.compute_grads(
                generated_image
            )

            if step is not None and (
                    iteration % step == 0 or iteration == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}".format(
                        iteration, total, content, style
                    )
                )

            if iteration != iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

            if total < best_cost:
                best_cost = total
                best_image = generated_image.numpy()

        return best_image[0], float(best_cost)
