"""
CNN-TCN-LSTM model for PPG/BVP-based stress/affect recognition.

Architecture reproduced from:
Alghoul et al. (2025), "Enhancing Generalization in PPG-Based Emotion
Recognition with a CNN-TCN-LSTM Model"

Input: raw PPG/BVP signal, windowed (default: 60s @ 64Hz = 3840 samples, 1 channel)
Output: 2-class softmax (binary stress/affect classification)

This is dataset-agnostic by design: it takes a (window_length, 1) input,
so it plugs in identically whether the upstream data is WESAD (64Hz wrist BVP),
PPGE (100Hz fingertip PPG, resampled), or a backup dataset like CLAS.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_branch(inputs):
    """Shared CNN feature extractor — two conv blocks per the paper."""
    # Conv1: 8 filters, kernel 64, stride 4, ReLU, same padding
    x = layers.Conv1D(filters=8, kernel_size=64, strides=4,
                       activation='relu', padding='same', name='cnn_conv1')(inputs)
    x = layers.MaxPooling1D(pool_size=2, name='cnn_pool1')(x)
    x = layers.BatchNormalization(name='cnn_bn1')(x)
    x = layers.Dropout(0.30, name='cnn_dropout1')(x)

    # Conv2: 16 filters, kernel 32, stride 2
    x = layers.Conv1D(filters=16, kernel_size=32, strides=2,
                       activation='relu', padding='same', name='cnn_conv2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='cnn_pool2')(x)
    x = layers.BatchNormalization(name='cnn_bn2')(x)
    x = layers.Dropout(0.30, name='cnn_dropout2')(x)

    return x


def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.30, name_prefix='tcn'):
    """
    One dilated causal-conv residual block with a skip connection.
    Two causal convs per block (standard TCN design), matching the paper's
    stated use of causal padding + skip connections.
    """
    prev = x

    conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding='causal',
                           activation='relu', name=f'{name_prefix}_conv1_d{dilation_rate}')(x)
    conv1 = layers.Dropout(dropout_rate, name=f'{name_prefix}_drop1_d{dilation_rate}')(conv1)

    conv2 = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding='causal',
                           activation='relu', name=f'{name_prefix}_conv2_d{dilation_rate}')(conv1)
    conv2 = layers.Dropout(dropout_rate, name=f'{name_prefix}_drop2_d{dilation_rate}')(conv2)

    # Skip connection: 1x1 conv on the residual path if channel dims differ
    if prev.shape[-1] != filters:
        prev = layers.Conv1D(filters=filters, kernel_size=1,
                              padding='same', name=f'{name_prefix}_skip_proj_d{dilation_rate}')(prev)

    out = layers.Add(name=f'{name_prefix}_residual_add_d{dilation_rate}')([prev, conv2])
    return out


def build_tcn_branch(inputs, filters=8, kernel_size=32, dilations=(1, 2, 4, 8),
                      dropout_rate=0.30, return_sequence=False):
    """
    TCN branch: dilations [1,2,4,8], causal padding, skip connections, 30% dropout.

    return_sequence=False (default, unchanged behavior): returns the pooled
        fixed-size representation, as used by the classifier in
        build_cnn_tcn_lstm_model().
    return_sequence=True: returns the pre-pooling (T', filters) sequence
        instead -- used by pretrain_encoder.py's decoder, which needs the
        temporal structure intact to reconstruct the full-length signal.
    """
    x = inputs
    for d in dilations:
        x = tcn_residual_block(x, filters=filters, kernel_size=kernel_size,
                                dilation_rate=d, dropout_rate=dropout_rate)
    if return_sequence:
        return x
    return layers.GlobalAveragePooling1D(name='tcn_global_pool')(x)


def build_lstm_branch(inputs, units=12):
    """LSTM branch: 12 units, per the paper."""
    x = layers.LSTM(units, name='lstm_branch')(inputs)
    return x


def build_cnn_tcn_lstm_model(window_length=3840, n_channels=1, n_classes=2,
                              tcn_filters=8, tcn_kernel_size=32,
                              tcn_dilations=(1, 2, 4, 8), lstm_units=12,
                              learning_rate=0.01):
    """
    Full model: shared CNN branch -> parallel TCN + LSTM branches -> concat -> softmax.

    Args:
        window_length: number of samples per window (default 3840 = 60s @ 64Hz)
        n_channels: input channels (1 for single-lead PPG/BVP)
        n_classes: output classes (2 for binary stress/affect)
    """
    inputs = layers.Input(shape=(window_length, n_channels), name='ppg_input')

    cnn_features = build_cnn_branch(inputs)

    tcn_out = build_tcn_branch(cnn_features, filters=tcn_filters,
                                kernel_size=tcn_kernel_size, dilations=tcn_dilations)
    lstm_out = build_lstm_branch(cnn_features, units=lstm_units)

    merged = layers.Concatenate(name='tcn_lstm_concat')([tcn_out, lstm_out])
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(merged)

    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_TCN_LSTM')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


if __name__ == '__main__':
    # Sanity check: build the model and run a dummy batch through it to confirm
    # shapes work end-to-end before any real data is available.
    import numpy as np

    model = build_cnn_tcn_lstm_model()
    model.summary()

    dummy_x = np.random.randn(4, 3840, 1).astype('float32')  # batch of 4 windows
    dummy_y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(4,)), num_classes=2)

    print("\n--- Forward pass sanity check ---")
    preds = model.predict(dummy_x, verbose=0)
    print("Output shape:", preds.shape, "(expected: (4, 2))")
    print("Sample output (should be valid softmax probs):", preds[0], "sum =", preds[0].sum())

    print("\n--- One training step sanity check ---")
    history = model.fit(dummy_x, dummy_y, epochs=1, batch_size=4, verbose=1)
    print("\nModel builds and trains end-to-end with dummy data. Ready to plug in real windows.")
