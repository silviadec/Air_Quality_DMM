"""Tests for model components."""

import pytest
import torch
from torch import Tensor

from dmm import (
    ModelConfig,
    GatedTransition,
    Emitter,
    Combiner,
    DeepMarkovModel,
    DMMFactory,
)


class TestGatedTransition:
    """Tests for GatedTransition component."""

    def test_output_shape(
        self, transition: GatedTransition, batch_latent_state: Tensor
    ) -> None:
        """Test that output has correct shape."""
        loc, scale = transition(batch_latent_state)

        batch_size = batch_latent_state.shape[0]
        z_dim = batch_latent_state.shape[1]

        assert loc.shape == (batch_size, z_dim)
        assert scale.shape == (batch_size, z_dim)

    def test_scale_positive(
        self, transition: GatedTransition, batch_latent_state: Tensor
    ) -> None:
        """Test that scale is always positive."""
        _, scale = transition(batch_latent_state)
        assert (scale > 0).all()

    def test_single_sample(
        self, transition: GatedTransition, latent_state: Tensor
    ) -> None:
        """Test with single (unbatched) input."""
        z = latent_state.unsqueeze(0)
        loc, scale = transition(z)

        assert loc.shape == (1, latent_state.shape[0])
        assert scale.shape == (1, latent_state.shape[0])

    def test_deterministic_output(
        self, transition: GatedTransition, batch_latent_state: Tensor
    ) -> None:
        """Test that output is deterministic for same input."""
        loc1, scale1 = transition(batch_latent_state)
        loc2, scale2 = transition(batch_latent_state)

        assert torch.allclose(loc1, loc2)
        assert torch.allclose(scale1, scale2)


class TestEmitter:
    """Tests for Emitter component."""

    def test_output_shape(
        self, emitter: Emitter, batch_latent_state: Tensor, model_config: ModelConfig
    ) -> None:
        """Test that output has correct shape."""
        loc, scale = emitter(batch_latent_state)

        batch_size = batch_latent_state.shape[0]
        output_dim = model_config.output_dim

        assert loc.shape == (batch_size, output_dim)
        assert scale.shape == (batch_size, output_dim)

    def test_scale_positive(
        self, emitter: Emitter, batch_latent_state: Tensor
    ) -> None:
        """Test that scale is always positive."""
        _, scale = emitter(batch_latent_state)
        assert (scale > 0).all()

    def test_output_dim_mapping(self, model_config: ModelConfig) -> None:
        """Test that emitter maps to correct output dimension."""
        emitter = Emitter(
            z_dim=model_config.z_dim,
            emission_dim=model_config.emission_dim,
            output_dim=7,  # Different output dim
        )

        z = torch.randn(4, model_config.z_dim)
        loc, scale = emitter(z)

        assert loc.shape == (4, 7)
        assert scale.shape == (4, 7)


class TestCombiner:
    """Tests for Combiner component."""

    def test_output_shape(
        self, combiner: Combiner, model_config: ModelConfig
    ) -> None:
        """Test that output has correct shape."""
        batch_size = 4
        z_prev = torch.randn(batch_size, model_config.z_dim)
        h_rnn = torch.randn(batch_size, model_config.rnn_dim)

        loc, scale = combiner(z_prev, h_rnn)

        assert loc.shape == (batch_size, model_config.z_dim)
        assert scale.shape == (batch_size, model_config.z_dim)

    def test_scale_positive(
        self, combiner: Combiner, model_config: ModelConfig
    ) -> None:
        """Test that scale is always positive."""
        z_prev = torch.randn(4, model_config.z_dim)
        h_rnn = torch.randn(4, model_config.rnn_dim)

        _, scale = combiner(z_prev, h_rnn)
        assert (scale > 0).all()

    def test_combines_both_inputs(
        self, combiner: Combiner, model_config: ModelConfig
    ) -> None:
        """Test that output changes when either input changes."""
        z_prev = torch.randn(4, model_config.z_dim)
        h_rnn = torch.randn(4, model_config.rnn_dim)

        loc1, _ = combiner(z_prev, h_rnn)

        # Change z_prev
        z_prev_new = torch.randn(4, model_config.z_dim)
        loc2, _ = combiner(z_prev_new, h_rnn)

        # Change h_rnn
        h_rnn_new = torch.randn(4, model_config.rnn_dim)
        loc3, _ = combiner(z_prev, h_rnn_new)

        assert not torch.allclose(loc1, loc2)
        assert not torch.allclose(loc1, loc3)


class TestDeepMarkovModel:
    """Tests for DeepMarkovModel."""

    def test_model_creation(self, model: DeepMarkovModel) -> None:
        """Test that model is created successfully."""
        assert model is not None
        assert hasattr(model, "trans")
        assert hasattr(model, "emitter")
        assert hasattr(model, "combiner")
        assert hasattr(model, "rnn")

    def test_model_attributes(
        self, model: DeepMarkovModel, model_config: ModelConfig
    ) -> None:
        """Test that model has correct attributes."""
        assert model.z_dim == model_config.z_dim
        assert model.rnn_dim == model_config.rnn_dim
        assert model.output_dim == model_config.output_dim

    def test_initial_z_shapes(
        self, model: DeepMarkovModel, model_config: ModelConfig
    ) -> None:
        """Test initial latent state shapes."""
        batch_size = 4
        z_model = model._get_initial_z_model(batch_size)
        z_guide = model._get_initial_z_guide(batch_size)

        assert z_model.shape == (batch_size, model_config.z_dim)
        assert z_guide.shape == (batch_size, model_config.z_dim)

    def test_encode_observations(
        self, model: DeepMarkovModel, batch_sequence: Tensor, model_config: ModelConfig
    ) -> None:
        """Test RNN encoding of observations."""
        batch_size = batch_sequence.shape[0]
        seq_length = batch_sequence.shape[1]

        rnn_output = model._encode_observations(batch_sequence, batch_size)

        assert rnn_output.shape == (batch_size, seq_length, model_config.rnn_dim)

    def test_get_z_0(self, model: DeepMarkovModel, model_config: ModelConfig) -> None:
        """Test getting initial latent state."""
        z_0 = model.get_z_0()
        assert z_0.shape == (model_config.z_dim,)


class TestDMMFactory:
    """Tests for DMMFactory."""

    def test_create_default(self, model_config: ModelConfig) -> None:
        """Test creating model with default components."""
        model = DMMFactory.create_default(model_config)

        assert isinstance(model, DeepMarkovModel)
        assert model.z_dim == model_config.z_dim

    def test_create_with_different_configs(self) -> None:
        """Test creating models with different configurations."""
        configs = [
            ModelConfig(z_dim=8, output_dim=3),
            ModelConfig(z_dim=16, output_dim=5),
            ModelConfig(z_dim=32, output_dim=10),
        ]

        for config in configs:
            model = DMMFactory.create_default(config)
            assert model.z_dim == config.z_dim
            assert model.output_dim == config.output_dim
