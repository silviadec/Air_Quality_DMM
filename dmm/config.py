"""Configuration dataclasses for the Deep Markov Model.

This module provides type-safe configuration through dataclasses for model
architecture, training hyperparameters, and data preprocessing settings.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Attributes:
        z_dim: Dimensionality of the latent state z_t.
        transition_dim: Hidden dimension for the transition network.
        emission_dim: Hidden dimension for the emission network.
        rnn_dim: Hidden dimension for the RNN in the guide.
        output_dim: Dimensionality of the observed variables x_t.
    """

    z_dim: int = 16
    transition_dim: int = 64
    emission_dim: int = 32
    rnn_dim: int = 64
    output_dim: int = 5


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.

    Attributes:
        lr: Learning rate for the optimizer.
        betas: Beta coefficients for Adam optimizer.
        clip_norm: Gradient clipping norm.
        num_epochs: Number of training epochs.
        print_every: Print training progress every N epochs.
        random_seed: Random seed for reproducibility.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    clip_norm: float = 5.0
    num_epochs: int = 2000
    print_every: int = 100
    random_seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data preprocessing.

    Attributes:
        max_nan_gap: Maximum consecutive NaN gap to interpolate.
        test_size: Fraction of sequences for test set.
        columns: List of column names to use as features.
    """

    max_nan_gap: int = 2
    test_size: float = 0.2
    columns: Tuple[str, ...] = ("PM10", "temp", "humidity", "rain", "wind")
