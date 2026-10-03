"""
ablation_models.py
==================

Publication-quality architecture ablation models for physiological stress recognition.
Investigates Research Question RQ-A:
Does the full CNN-TCN-LSTM architecture provide a meaningful advantage over simpler
temporal architectures for physiological signal classification?

Ablation Models:
    A1: CNN only        - Shared CNN feature extractor -> GlobalAveragePooling1D -> Dense(2, softmax)
    A2: CNN + LSTM      - Shared CNN feature extractor -> LSTM branch -> Dense(2, softmax)
    A3: CNN + TCN       - Shared CNN feature extractor -> TCN branch (causal dilated convs) -> Dense(2, softmax)
    A4: CNN + TCN + LSTM- Full model from `src/model.py` (shared CNN -> parallel TCN + LSTM -> concat -> Dense(2, softmax))

All models reuse the exact same locked CNN, TCN, and LSTM components from `src/model.py`.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from model import (
    build_cnn_branch,
    build_tcn_branch,
    build_lstm_branch,
    build_cnn_tcn_lstm_model,
)


def build_cnn_only_model(
    window_length: int = 3840,
    n_channels: int = 1,
    n_classes: int = 2,
    learning_rate: float = 0.01,
) -> models.Model:
    """
    A1: CNN Only Architecture.
    Applies the locked 2-stage Conv1D feature extractor followed by
    GlobalAveragePooling1D to obtain a fixed-size representation, then Dense(2, softmax).
    """
    inputs = layers.Input(shape=(window_length, n_channels), name="ppg_input")
    cnn_features = build_cnn_branch(inputs)
    pooled = layers.GlobalAveragePooling1D(name="cnn_global_pool")(cnn_features)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(pooled)

    model = models.Model(inputs=inputs, outputs=outputs, name="A1_CNN_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_cnn_lstm_model(
    window_length: int = 3840,
    n_channels: int = 1,
    n_classes: int = 2,
    lstm_units: int = 12,
    learning_rate: float = 0.01,
) -> models.Model:
    """
    A2: CNN + LSTM Architecture.
    Feeds the locked CNN feature representation into the locked 12-unit LSTM branch,
    followed by Dense(2, softmax).
    """
    inputs = layers.Input(shape=(window_length, n_channels), name="ppg_input")
    cnn_features = build_cnn_branch(inputs)
    lstm_out = build_lstm_branch(cnn_features, units=lstm_units)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(lstm_out)

    model = models.Model(inputs=inputs, outputs=outputs, name="A2_CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_cnn_tcn_model(
    window_length: int = 3840,
    n_channels: int = 1,
    n_classes: int = 2,
    tcn_filters: int = 8,
    tcn_kernel_size: int = 32,
    tcn_dilations: tuple = (1, 2, 4, 8),
    learning_rate: float = 0.01,
) -> models.Model:
    """
    A3: CNN + TCN Architecture.
    Feeds the locked CNN feature representation into the locked TCN branch
    (causal dilated convs [1,2,4,8] with skip connections and GlobalAveragePooling1D),
    followed by Dense(2, softmax).
    """
    inputs = layers.Input(shape=(window_length, n_channels), name="ppg_input")
    cnn_features = build_cnn_branch(inputs)
    tcn_out = build_tcn_branch(
        cnn_features,
        filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        dilations=tcn_dilations,
    )
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(tcn_out)

    model = models.Model(inputs=inputs, outputs=outputs, name="A3_CNN_TCN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_full_cnn_tcn_lstm_model(
    window_length: int = 3840,
    n_channels: int = 1,
    n_classes: int = 2,
    learning_rate: float = 0.01,
) -> models.Model:
    """
    A4: CNN + TCN + LSTM Architecture.
    Directly invokes the locked reference implementation from `src/model.py`.
    Exposes `tcn_lstm_concat` layer output for downstream latent representation.
    """
    return build_cnn_tcn_lstm_model(
        window_length=window_length,
        n_channels=n_channels,
        n_classes=n_classes,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    print("--- ablation_models.py sanity test ---")
    dummy = tf.zeros((2, 3840, 1), dtype=tf.float32)

    models_dict = {
        "A1: CNN only": build_cnn_only_model(),
        "A2: CNN + LSTM": build_cnn_lstm_model(),
        "A3: CNN + TCN": build_cnn_tcn_model(),
        "A4: CNN + TCN + LSTM": build_full_cnn_tcn_lstm_model(),
    }

    for name, m in models_dict.items():
        out = m(dummy)
        params = m.count_params()
        print(f"{name:22s} | Input: {m.input_shape} | Output: {out.shape} | Params: {params:,}")
        assert out.shape == (2, 2), f"Unexpected output shape {out.shape} for {name}"

    print("\nAll 4 ablation architectures compiled and verified successfully.")
