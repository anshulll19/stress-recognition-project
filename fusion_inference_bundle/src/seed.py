"""
# seed.py

Reproducibility utility for controlling random seeds across Python, NumPy,
and TensorFlow.

Usage:
    from seed import set_seed
    set_seed(42)

Note on Determinism:
Setting random seeds guarantees reproducible initializations, NumPy array operations,
and standard TensorFlow shuffle operations. However, bit-for-bit deterministic GPU
behavior cannot be strictly guaranteed across different hardware/CUDA/CuDNN configurations
due to non-deterministic atomic operations and floating-point reduction order in certain
convolutions and cuDNN LSTM backpropagation passes.
"""

import os
import random
import numpy as np
import tensorflow as tf

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    """
    Sets random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value (default: 42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    print("--- seed.py self-test ---")
    set_seed(42)
    r1 = random.random()
    np1 = np.random.rand(3)
    tf1 = tf.random.uniform([3]).numpy()

    set_seed(42)
    r2 = random.random()
    np2 = np.random.rand(3)
    tf2 = tf.random.uniform([3]).numpy()

    assert r1 == r2, "Python random seed failed"
    assert np.allclose(np1, np2), "NumPy random seed failed"
    assert np.allclose(tf1, tf2), "TensorFlow random seed failed"

    print(f"Seed verified: Python ({r1:.4f}), NumPy ({np1[0]:.4f}), TF ({tf1[0]:.4f}) match.")
    print("Self-test passed.")
