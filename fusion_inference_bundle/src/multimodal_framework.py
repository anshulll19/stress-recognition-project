"""
multimodal_framework.py
=======================

Framework and model architecture for future EmpathicSchool multimodal experiments (E5 - E9).

DESIGN PRINCIPLES:
1. Controlled Comparison (Scientific Rigor):
   - Compares:
       (A) Target-only physiological encoder
       (B) Cross-dataset-pretrained physiological encoder (from WESAD + CLAS)
     under the EXACT SAME multimodal architecture, target subject splits,
     video features, optimizer hyperparameters, and evaluation metrics.
2. Architecture Preservation:
   - The physiological encoder strictly reuses the CNN-TCN-LSTM architecture from `model.py`.
   - Feature representations are tapped directly from the pre-classification concatenation layer ('tcn_lstm_concat').
3. Strict Subject-Level Leakage Prevention:
   - Train, validation, and test splits are strictly partitioned by subject.
   - Video and physiological streams from the same subject always reside in the same partition.

EXPERIMENT TAXONOMY:
    E1_wesad_loso:                           WESAD intra-dataset LOSO benchmark
    E2_wesad_to_clas:                        Cross-dataset WESAD -> CLAS
    E3_clas_to_wesad:                        Cross-dataset CLAS -> WESAD
    E4_wesad_clas_to_target:                 Pooled source pretraining
    E5_empathicschool_physiology_only:       Target unimodal physiology baseline
    E6_cross_dataset_pretrained_multimodal:  Multimodal with cross-dataset pretrained physio encoder
    E7_empathicschool_video_only:            Target unimodal video baseline
    E8_empathicschool_target_only_multimodal:Multimodal with scratch/target-only physio encoder
    E9_empathicschool_fusion_comparison:     Comparison across early/late multimodal fusion strategies
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras import layers, models

from model import build_cnn_tcn_lstm_model
from seed import set_seed
from evaluation import compute_metrics, log_experiment


def split_by_subject(
    subject_ids: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions indices into Train, Validation, and Test strictly at the subject level.

    Guarantees:
        Subjects(Train) ∩ Subjects(Validation) = ∅
        Subjects(Train) ∩ Subjects(Test) = ∅
        Subjects(Validation) ∩ Subjects(Test) = ∅

    Returns:
        (train_idx, val_idx, test_idx) as 1D integer index arrays.
    """
    subjects = np.asarray(subject_ids)
    all_indices = np.arange(len(subjects))

    # Split off test set by subject
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    dev_idx, test_idx = next(gss_test.split(all_indices, groups=subjects))

    # Split remaining dev set into train and validation by subject
    dev_subjects = subjects[dev_idx]
    dev_val_fraction = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=dev_val_fraction, random_state=random_seed)
    dev_train_subidx, dev_val_subidx = next(gss_val.split(dev_idx, groups=dev_subjects))

    train_idx = dev_idx[dev_train_subidx]
    val_idx = dev_idx[dev_val_subidx]

    # Verification of zero subject leakage
    train_subs = set(subjects[train_idx])
    val_subs = set(subjects[val_idx])
    test_subs = set(subjects[test_idx])

    assert not (train_subs & val_subs), f"Leakage detected between train and val: {train_subs & val_subs}"
    assert not (train_subs & test_subs), f"Leakage detected between train and test: {train_subs & test_subs}"
    assert not (val_subs & test_subs), f"Leakage detected between val and test: {val_subs & test_subs}"

    return train_idx, val_idx, test_idx


def extract_physio_feature_extractor(
    base_model: models.Model,
    freeze_weights: bool = False,
    name: str = "physio_feature_extractor",
) -> models.Model:
    """
    Taps the latent representation ('tcn_lstm_concat') of a compiled CNN-TCN-LSTM model.
    Preserves exact weights, layers, and configuration without modification.
    """
    latent_output = base_model.get_layer("tcn_lstm_concat").output
    extractor = models.Model(inputs=base_model.input, outputs=latent_output, name=name)

    if freeze_weights:
        for layer in extractor.layers:
            layer.trainable = False

    return extractor


def build_video_feature_branch(
    video_input_shape: Tuple[int, ...],
    projection_units: int = 32,
    name: str = "video_branch",
) -> models.Model:
    """
    Standard projection head for pre-extracted or temporal video features.
    Configurable to match input feature dimensionality.
    """
    video_inputs = layers.Input(shape=video_input_shape, name="video_input")

    if len(video_input_shape) == 1:
        # Pre-pooled feature vector per window (e.g. D-dimensional embedding)
        x = layers.Dense(projection_units, activation="relu", name=f"{name}_proj")(video_inputs)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Dropout(0.3, name=f"{name}_dropout")(x)
    elif len(video_input_shape) == 2:
        # Sequence of frame features: (T_frames, D_features)
        x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(video_inputs)
        x = layers.Dense(projection_units, activation="relu", name=f"{name}_proj")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Dropout(0.3, name=f"{name}_dropout")(x)
    else:
        raise ValueError(f"Unsupported video feature shape: {video_input_shape}")

    return models.Model(inputs=video_inputs, outputs=x, name=name)


def build_controlled_multimodal_model(
    physio_encoder: models.Model,
    video_input_shape: Tuple[int, ...],
    fusion_units: int = 32,
    n_classes: int = 2,
    learning_rate: float = 0.01,
    name: str = "Multimodal_Fusion_Model",
) -> models.Model:
    """
    Builds the combined multimodal architecture.

    CONTROLLED EXPERIMENTAL COMPARISON:
    - Pass in an uninitialized (scratch) physio_encoder -> Model A (E8: Target-Only Physio)
    - Pass in a cross-dataset pretrained physio_encoder -> Model B (E6: Pretrained Physio)
    All other heads, inputs, and optimizers remain strictly identical.
    """
    video_branch = build_video_feature_branch(video_input_shape, projection_units=fusion_units)

    physio_in = physio_encoder.input
    physio_feat = physio_encoder.output

    video_in = video_branch.input
    video_feat = video_branch.output

    # Intermediate Multimodal Fusion Head
    fused = layers.Concatenate(name="fusion_concat")([physio_feat, video_feat])
    fused = layers.Dense(fusion_units, activation="relu", name="fusion_dense")(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    fused = layers.Dropout(0.3, name="fusion_dropout")(fused)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(fused)

    model = models.Model(inputs=[physio_in, video_in], outputs=outputs, name=name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def build_video_only_model(
    video_input_shape: Tuple[int, ...],
    hidden_units: int = 32,
    n_classes: int = 2,
    learning_rate: float = 0.01,
) -> models.Model:
    """Video-only baseline architecture (E7)."""
    video_branch = build_video_feature_branch(video_input_shape, projection_units=hidden_units)
    x = video_branch.output
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=video_branch.input, outputs=outputs, name="Video_Only_Baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    print("--- multimodal_framework self-test ---")
    set_seed(42)

    # 1. Test subject-level splitting with zero leakage
    print("1. Testing subject-level splitting...")
    fake_subjects = np.repeat([f"Subject_{i}" for i in range(10)], 20)
    train_idx, val_idx, test_idx = split_by_subject(fake_subjects, test_size=0.2, val_size=0.2, random_seed=42)
    print(f"   Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    print("   No subject leakage verified successfully.")

    # 2. Test physiological feature extractor extraction
    print("\n2. Testing physiological feature extractor from CNN-TCN-LSTM...")
    base_model = build_cnn_tcn_lstm_model()
    physio_extractor = extract_physio_feature_extractor(base_model)
    dummy_physio = tf.zeros((2, 3840, 1))
    extracted_features = physio_extractor(dummy_physio)
    print(f"   Physio latent representation shape: {extracted_features.shape}")

    # 3. Test controlled multimodal model compilation
    print("\n3. Testing controlled multimodal model assembly...")
    video_shape = (64,)  # e.g., 64-dimensional feature vector per window
    multimodal_model = build_controlled_multimodal_model(physio_extractor, video_input_shape=video_shape)
    dummy_video = tf.zeros((2, 64))
    preds = multimodal_model([dummy_physio, dummy_video])
    print(f"   Multimodal forward pass output shape: {preds.shape}")

    # 4. Test video-only baseline compilation
    print("\n4. Testing video-only baseline assembly...")
    video_only = build_video_only_model(video_input_shape=video_shape)
    preds_video = video_only(dummy_video)
    print(f"   Video-only forward pass output shape: {preds_video.shape}")

    print("\nMultimodal experiment architecture validated successfully.")
