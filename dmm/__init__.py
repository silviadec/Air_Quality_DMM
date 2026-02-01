"""Deep Markov Model package.

A modular implementation of Deep Markov Models for probabilistic
time series modeling, following SOLID principles.

Example usage:
    >>> from dmm import ModelConfig, DMMFactory, DMMTrainer
    >>>
    >>> # Create model with default components
    >>> config = ModelConfig(z_dim=16, output_dim=5)
    >>> model = DMMFactory.create_default(config)
    >>>
    >>> # Train the model
    >>> trainer = DMMTrainer(model)
    >>> history = trainer.fit(train_seqs, test_seqs)
"""

from .config import ModelConfig, TrainingConfig, DataConfig
from .models import (
    BaseTransition,
    BaseEmitter,
    BaseCombiner,
    GatedTransition,
    Emitter,
    Combiner,
    DeepMarkovModel,
    DMMFactory,
)
from .training import DMMTrainer
from .inference import DMMForecaster
from .analysis import DMMInterpreter
from .data import (
    zscore_normalize,
    fill_short_nan_intervals,
    preprocess_dataframe,
    nan_summary,
    NormalizationStats,
    build_sequences,
    sequences_to_tensors,
    train_test_split_sequences,
    SequenceDataset,
)
from .visualization import (
    plot_time_series,
    plot_correlation_matrix,
    plot_nan_calendar,
    plot_nan_context,
    plot_training_loss,
    plot_forecast,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # Models
    "BaseTransition",
    "BaseEmitter",
    "BaseCombiner",
    "GatedTransition",
    "Emitter",
    "Combiner",
    "DeepMarkovModel",
    "DMMFactory",
    # Training
    "DMMTrainer",
    # Inference
    "DMMForecaster",
    # Analysis
    "DMMInterpreter",
    # Data
    "zscore_normalize",
    "fill_short_nan_intervals",
    "preprocess_dataframe",
    "nan_summary",
    "NormalizationStats",
    "build_sequences",
    "sequences_to_tensors",
    "train_test_split_sequences",
    "SequenceDataset",
    # Visualization
    "plot_time_series",
    "plot_correlation_matrix",
    "plot_nan_calendar",
    "plot_nan_context",
    "plot_training_loss",
    "plot_forecast",
]
