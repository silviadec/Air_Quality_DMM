"""Tests for inference module."""

from typing import List

import pytest
import torch
from torch import Tensor

from dmm import (
    ModelConfig,
    DeepMarkovModel,
    DMMForecaster,
    DMMFactory,
)


class TestDMMForecaster:
    """Tests for DMMForecaster."""

    def test_forecaster_creation(self, model: DeepMarkovModel) -> None:
        """Test that forecaster is created successfully."""
        forecaster = DMMForecaster(model)

        assert forecaster.model is model

    def test_infer_latent(
        self,
        model: DeepMarkovModel,
        sample_sequence: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test latent state inference."""
        forecaster = DMMForecaster(model)

        z_last = forecaster.infer_latent(sample_sequence)

        assert z_last.shape == (model_config.z_dim,)
        assert not torch.isnan(z_last).any()

    def test_infer_latent_sample(
        self,
        model: DeepMarkovModel,
        sample_sequence: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test latent state sampling."""
        forecaster = DMMForecaster(model)
        num_samples = 10

        z_samples = forecaster.infer_latent_sample(sample_sequence, num_samples)

        assert z_samples.shape == (num_samples, model_config.z_dim)
        assert not torch.isnan(z_samples).any()

    def test_forecast(
        self,
        model: DeepMarkovModel,
        latent_state: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test forecasting from latent state."""
        forecaster = DMMForecaster(model)
        steps = 5
        num_samples = 10

        predictions = forecaster.forecast(
            z_last=latent_state,
            steps=steps,
            num_samples=num_samples,
        )

        assert predictions.shape == (num_samples, steps, model_config.output_dim)
        assert not torch.isnan(predictions).any()

    def test_forecast_batched_input(
        self,
        model: DeepMarkovModel,
        batch_latent_state: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test forecasting with batched latent state."""
        forecaster = DMMForecaster(model)
        steps = 5
        num_samples = 10

        # Use first sample from batch
        z_last = batch_latent_state[0]

        predictions = forecaster.forecast(
            z_last=z_last,
            steps=steps,
            num_samples=num_samples,
        )

        assert predictions.shape == (num_samples, steps, model_config.output_dim)

    def test_forecast_mean(
        self,
        model: DeepMarkovModel,
        latent_state: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test deterministic mean forecasting."""
        forecaster = DMMForecaster(model)
        steps = 5

        predictions = forecaster.forecast_mean(
            z_last=latent_state,
            steps=steps,
        )

        assert predictions.shape == (steps, model_config.output_dim)
        assert not torch.isnan(predictions).any()

    def test_forecast_mean_deterministic(
        self,
        model: DeepMarkovModel,
        latent_state: Tensor,
    ) -> None:
        """Test that mean forecast is deterministic."""
        forecaster = DMMForecaster(model)
        steps = 5

        pred1 = forecaster.forecast_mean(z_last=latent_state, steps=steps)
        pred2 = forecaster.forecast_mean(z_last=latent_state, steps=steps)

        assert torch.allclose(pred1, pred2)

    def test_get_prediction_intervals(
        self,
        model: DeepMarkovModel,
        latent_state: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test prediction interval computation."""
        forecaster = DMMForecaster(model)
        steps = 5
        num_samples = 100

        predictions = forecaster.forecast(
            z_last=latent_state,
            steps=steps,
            num_samples=num_samples,
        )

        mean, lower, upper = forecaster.get_prediction_intervals(
            predictions,
            lower_quantile=0.1,
            upper_quantile=0.9,
        )

        assert mean.shape == (steps, model_config.output_dim)
        assert lower.shape == (steps, model_config.output_dim)
        assert upper.shape == (steps, model_config.output_dim)

        # Lower should be less than or equal to mean, mean less than or equal to upper
        assert (lower <= mean + 1e-6).all()
        assert (mean <= upper + 1e-6).all()

    def test_end_to_end_forecast(
        self,
        model: DeepMarkovModel,
        sample_sequence: Tensor,
        model_config: ModelConfig,
    ) -> None:
        """Test complete inference and forecast pipeline."""
        forecaster = DMMForecaster(model)
        steps = 10
        num_samples = 50

        # Infer latent state from observed sequence
        z_last = forecaster.infer_latent(sample_sequence)

        # Generate forecasts
        predictions = forecaster.forecast(
            z_last=z_last,
            steps=steps,
            num_samples=num_samples,
        )

        # Get prediction intervals
        mean, lower, upper = forecaster.get_prediction_intervals(predictions)

        # Verify shapes
        assert z_last.shape == (model_config.z_dim,)
        assert predictions.shape == (num_samples, steps, model_config.output_dim)
        assert mean.shape == (steps, model_config.output_dim)

    def test_forecast_variability(
        self,
        model: DeepMarkovModel,
        latent_state: Tensor,
    ) -> None:
        """Test that stochastic forecast produces variable outputs."""
        forecaster = DMMForecaster(model)
        steps = 5
        num_samples = 50

        predictions = forecaster.forecast(
            z_last=latent_state,
            steps=steps,
            num_samples=num_samples,
        )

        # Check that samples are not all identical (model is stochastic)
        std = predictions.std(dim=0)
        assert (std > 0).any(), "Forecast samples should have some variability"
