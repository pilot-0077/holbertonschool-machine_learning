#!/usr/bin/env python3
"""Trains a CIFAR-10 classifier using transfer learning."""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 images and labels for the model.

    Args:
        X: NumPy array of shape (m, 32, 32, 3) containing images.
        Y: NumPy array of shape (m,) or (m, 1) containing labels.

    Returns:
        X_p: Preprocessed image data.
        Y_p: One-hot encoded labels.
    """
    X_p = X.astype('float32')
    X_p = K.applications.mobilenet_v2.preprocess_input(X_p)
    Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


def resize_images(images):
    """Resize CIFAR-10 images from 32x32 to 160x160."""
    return K.backend.resize_images(
        images,
        height_factor=5,
        width_factor=5,
        data_format='channels_last',
        interpolation='bilinear'
    )


def build_base_model():
    """Build the frozen MobileNetV2 feature extractor."""
    base_model = K.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3),
        pooling='avg'
    )
    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    resized = K.layers.Lambda(resize_images, name='resize')(inputs)
    features = base_model(resized, training=False)
    extractor = K.models.Model(inputs=inputs, outputs=features)
    return extractor, base_model


def build_classifier(feature_size):
    """Build and compile the CIFAR-10 classification head."""
    inputs = K.Input(shape=(feature_size,))
    x = K.layers.Dense(
        512,
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(inputs)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    classifier = K.models.Model(inputs=inputs, outputs=outputs)
    classifier.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return classifier


def train_model():
    """Train, fine-tune, evaluate, and save the CIFAR-10 model."""
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    X_fit = X_train[:45000]
    Y_fit = Y_train[:45000]
    X_valid = X_train[45000:]
    Y_valid = Y_train[45000:]

    extractor, base_model = build_base_model()

    train_features = extractor.predict(
        X_fit,
        batch_size=128,
        verbose=1
    )
    valid_features = extractor.predict(
        X_valid,
        batch_size=128,
        verbose=1
    )

    classifier = build_classifier(train_features.shape[-1])
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]

    classifier.fit(
        train_features,
        Y_fit,
        validation_data=(valid_features, Y_valid),
        epochs=25,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    inputs = K.Input(shape=(32, 32, 3))
    resized = K.layers.Lambda(resize_images, name='resize')(inputs)
    features = base_model(resized, training=False)
    outputs = classifier(features)
    model = K.models.Model(inputs=inputs, outputs=outputs)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        if isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=1,
            min_lr=1e-7
        )
    ]

    model.fit(
        X_fit,
        Y_fit,
        validation_data=(X_valid, Y_valid),
        epochs=8,
        batch_size=64,
        callbacks=fine_tune_callbacks,
        verbose=1
    )

    loss, accuracy = model.evaluate(
        X_test,
        Y_test,
        batch_size=128,
        verbose=1
    )
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    model.save('cifar10.h5')


if __name__ == '__main__':
    train_model()
