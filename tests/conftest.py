"""Pytest fixtures for DMM tests."""

from typing import List

import pytest
import torch
from torch import Tensor

from dmm import (
    ModelConfig,
    TrainingConfig,
    DMMFactory,
    DeepMarkovModel,
    GatedTransition,
    Emitter,
    Combiner,
)


@pytest.fixture
def model_config() -> ModelConfig:
    """Create a ModelConfig for testing with small dimensions."""
    return ModelConfig(
        z_dim=8,
        transition_dim=16,
        emission_dim=16,
        rnn_dim=16,
        output_dim=3,
    )


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create a TrainingConfig for testing with minimal epochs."""
    return TrainingConfig(
        lr=1e-3,
        clip_norm=5.0,
        num_epochs=2,
        print_every=1,
        random_seed=42,
    )


@pytest.fixture
def model(model_config: ModelConfig) -> DeepMarkovModel:
    """Create a DeepMarkovModel instance for testing."""
    return DMMFactory.create_default(model_config)


@pytest.fixture
def transition(model_config: ModelConfig) -> GatedTransition:
    """Create a GatedTransition component for testing."""
    return GatedTransition(
        z_dim=model_config.z_dim,
        transition_dim=model_config.transition_dim,
    )


@pytest.fixture
def emitter(model_config: ModelConfig) -> Emitter:
    """Create an Emitter component for testing."""
    return Emitter(
        z_dim=model_config.z_dim,
        emission_dim=model_config.emission_dim,
        output_dim=model_config.output_dim,
    )


@pytest.fixture
def combiner(model_config: ModelConfig) -> Combiner:
    """Create a Combiner component for testing."""
    return Combiner(
        z_dim=model_config.z_dim,
        rnn_dim=model_config.rnn_dim,
    )


@pytest.fixture
def sample_sequence(model_config: ModelConfig) -> Tensor:
    """Create a sample sequence tensor for testing."""
    torch.manual_seed(42)
    seq_length = 10
    return torch.randn(seq_length, model_config.output_dim)


@pytest.fixture
def sample_sequences(model_config: ModelConfig) -> List[Tensor]:
    """Create a list of sample sequence tensors for testing."""
    torch.manual_seed(42)
    sequences = []
    for i in range(5):
        seq_length = 8 + i * 2  # Variable lengths: 8, 10, 12, 14, 16
        sequences.append(torch.randn(seq_length, model_config.output_dim))
    return sequences


@pytest.fixture
def batch_sequence(model_config: ModelConfig) -> Tensor:
    """Create a batched sequence tensor for testing."""
    torch.manual_seed(42)
    batch_size = 4
    seq_length = 10
    return torch.randn(batch_size, seq_length, model_config.output_dim)


@pytest.fixture
def latent_state(model_config: ModelConfig) -> Tensor:
    """Create a sample latent state tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(model_config.z_dim)


@pytest.fixture
def batch_latent_state(model_config: ModelConfig) -> Tensor:
    """Create a batched latent state tensor for testing."""
    torch.manual_seed(42)
    batch_size = 4
    return torch.randn(batch_size, model_config.z_dim)
