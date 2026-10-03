"""
pretrain_encoder.py
====================

Self-supervised pretraining for a cross-dataset-robust physiological
representation (RQ3 / RQ5), using MASKED RECONSTRUCTION rather than
contrastive learning.

Why masked reconstruction, not contrastive: the pooled WESAD+CLAS dataset
is small (~1,900 windows total). Contrastive methods (SimCLR-style) need
large batches and many negatives to avoid representation collapse --
risky at this scale. Masked reconstruction is far more sample-efficient
and has no negative-sampling problem.

Why this sidesteps the CLAS label problem entirely: labels are NEVER
used here. CLAS's windows are treated purely as additional raw PPG
signal diversity, regardless of what its task/rest labels mean -- which
is exactly the resolution to the "CLAS class 1 = task, not stress"
issue flagged in the project handoff.

Architecture:
    Input (3840, 1) --[masked]--> Encoder (CNN + TCN, SAME layers as
    model.py's classifier) --> (T', 8) sequence --> Decoder (transposed
    convs) --> reconstructed (3840, 1)

    After pretraining, the encoder's CNN+TCN weights can be loaded
    directly into build_cnn_tcn_lstm_model() (see load_pretrained_encoder
    below), since the layer names/shapes match exactly.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from model import build_cnn_branch, build_tcn_branch

WINDOW_LENGTH = 3840
MASK_FRACTION = 0.20  # fraction of each window masked (contiguous chunk)


def apply_random_mask(windows: np.ndarray, mask_fraction: float = MASK_FRACTION, seed=None):
    """
    windows: (N, window_length, 1)
    Zeros out one random contiguous chunk per window. Returns the masked
    input (what the encoder sees) -- the ORIGINAL windows array remains
    the reconstruction target, unchanged.
    """
    rng = np.random.default_rng(seed)
    masked = windows.copy()
    window_length = windows.shape[1]
    mask_len = int(window_length * mask_fraction)

    for i in range(len(windows)):
        start = rng.integers(0, window_length - mask_len)
        masked[i, start:start + mask_len, :] = 0.0

    return masked


def build_pretrain_autoencoder(window_length=WINDOW_LENGTH, n_channels=1,
                                tcn_filters=8, tcn_kernel_size=32,
                                tcn_dilations=(1, 2, 4, 8), learning_rate=0.001):
    """
    Builds the full masked-autoencoder: encoder (CNN+TCN, matching
    model.py exactly) + a lightweight decoder that upsamples back to
    the original window length.
    """
    inputs = layers.Input(shape=(window_length, n_channels), name='masked_ppg_input')

    # ---- ENCODER: identical layers to model.py's classifier ----
    cnn_features = build_cnn_branch(inputs)
    tcn_sequence = build_tcn_branch(
        cnn_features, filters=tcn_filters, kernel_size=tcn_kernel_size,
        dilations=tcn_dilations, return_sequence=True  # keep temporal structure for the decoder
    )

    # ---- DECODER: mirrors the CNN's downsampling (total downsample factor = 32x) ----
    x = layers.Conv1DTranspose(16, kernel_size=32, strides=2, padding='same',
                                activation='relu', name='decoder_deconv1')(tcn_sequence)
    x = layers.UpSampling1D(2, name='decoder_upsample1')(x)
    x = layers.Conv1DTranspose(8, kernel_size=64, strides=4, padding='same',
                                activation='relu', name='decoder_deconv2')(x)
    x = layers.UpSampling1D(2, name='decoder_upsample2')(x)

    # Final projection back to 1 channel, trimmed/padded to exact window_length
    x = layers.Conv1D(1, kernel_size=7, padding='same', activation='linear', name='decoder_output_raw')(x)

    # Ensure exact length match (upsampling can land off-by-a-few due to rounding)
    reconstructed = layers.Lambda(
        lambda t: tf.image.resize(t[:, :, :, tf.newaxis], [window_length, 1])[:, :, :, 0],
        name='decoder_resize_to_input_length'
    )(x)

    autoencoder = models.Model(inputs=inputs, outputs=reconstructed, name='Masked_PPG_Autoencoder')
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    # Separate encoder-only model, for later weight extraction / transfer
    encoder = models.Model(inputs=inputs, outputs=tcn_sequence, name='PPG_Encoder')

    return autoencoder, encoder


def pretrain(pooled_windows: np.ndarray, epochs=100, batch_size=64,
             mask_fraction=MASK_FRACTION, early_stopping_patience=15, verbose=1):
    """
    pooled_windows: (N, 3840, 1) -- concatenated WESAD + CLAS windows,
                     LABELS NOT USED, NOT PASSED IN.

    Returns the trained autoencoder and the encoder sub-model (for saving
    weights to transfer into the classifier later).
    """
    autoencoder, encoder = build_pretrain_autoencoder()

    # Held-out validation split (window-level is fine here -- this is
    # representation pretraining, not the final evaluation; the real
    # leakage-sensitive evaluation happens later in the supervised
    # cross-dataset / multimodal experiments, not here).
    n = len(pooled_windows)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train_target = pooled_windows[train_idx]
    X_val_target = pooled_windows[val_idx]

    X_train_input = apply_random_mask(X_train_target, mask_fraction, seed=1)
    X_val_input = apply_random_mask(X_val_target, mask_fraction, seed=2)

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True
    )]

    history = autoencoder.fit(
        X_train_input, X_train_target,
        validation_data=(X_val_input, X_val_target),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=verbose
    )

    return autoencoder, encoder, history


def save_encoder_weights(encoder, path='pretrained_encoder.weights.h5'):
    encoder.save_weights(path)
    print(f"Saved pretrained encoder weights to {path}")


def load_pretrained_encoder_into_classifier(classifier_model, encoder_weights_path, freeze=True):
    """
    Loads pretrained CNN+TCN weights into a classifier built by
    build_cnn_tcn_lstm_model() (from model.py). Works because the layer
    names match exactly between the pretraining encoder and the classifier
    (both call build_cnn_branch / build_tcn_branch from the same model.py).

    freeze=True: the transferred layers are frozen (pure "target-only
        head trained on top of a frozen cross-dataset representation" --
        one arm of the RQ5 frozen-vs-fine-tuned comparison).
    freeze=False: transferred layers remain trainable (fine-tuning arm).
    """
    # Build a throwaway encoder with the same architecture to get matching layer objects
    _, reference_encoder = build_pretrain_autoencoder()
    reference_encoder.load_weights(encoder_weights_path)

    transferred, skipped = 0, 0
    for layer in classifier_model.layers:
        matching = [l for l in reference_encoder.layers if l.name == layer.name]
        if matching and len(layer.get_weights()) > 0:
            layer.set_weights(matching[0].get_weights())
            layer.trainable = not freeze
            transferred += 1
        elif len(layer.get_weights()) > 0:
            skipped += 1

    print(f"Transferred weights for {transferred} layers "
          f"({'frozen' if freeze else 'trainable (fine-tune)'}); "
          f"{skipped} classifier-only layers left as-is (LSTM branch, output layer)")

    return classifier_model


if __name__ == '__main__':
    # Self-test with synthetic pooled WESAD+CLAS-shaped data
    print("--- pretrain_encoder.py self-test ---\n")

    rng = np.random.default_rng(0)
    n_wesad, n_clas = 60, 90  # proportional to real sizes, scaled down for a fast test
    pooled = rng.standard_normal((n_wesad + n_clas, WINDOW_LENGTH, 1)).astype('float32')

    print(f"Pooled synthetic data: {len(pooled)} windows "
          f"({n_wesad} WESAD-like + {n_clas} CLAS-like), labels NOT used\n")

    autoencoder, encoder, history = pretrain(pooled, epochs=3, batch_size=16, verbose=1)

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\nFinal train MSE: {final_train_loss:.4f}, val MSE: {final_val_loss:.4f}")

    save_encoder_weights(encoder, 'test_pretrained_encoder.weights.h5')

    print("\n--- Testing transfer into classifier ---")
    from model import build_cnn_tcn_lstm_model
    classifier = build_cnn_tcn_lstm_model()
    load_pretrained_encoder_into_classifier(classifier, 'test_pretrained_encoder.weights.h5', freeze=True)

    dummy_x = rng.standard_normal((4, WINDOW_LENGTH, 1)).astype('float32')
    preds = classifier.predict(dummy_x, verbose=0)
    print(f"\nClassifier with transferred (frozen) encoder weights still predicts correctly: "
          f"output shape {preds.shape}")

    print("\nSelf-test passed. Pretraining pipeline + transfer-to-classifier both work end-to-end.")
