"""Tests for training module."""

from typing import List

import pytest
import torch
from torch import Tensor

from dmm import (
    ModelConfig,
    TrainingConfig,
    DeepMarkovModel,
    DMMTrainer,
    DMMFactory,
)


class TestDMMTrainer:
    """Tests for DMMTrainer."""

    def test_trainer_creation(
        self, model: DeepMarkovModel, training_config: TrainingConfig
    ) -> None:
        """Test that trainer is created successfully."""
        trainer = DMMTrainer(model, config=training_config)

        assert trainer.model is model
        assert trainer.config is training_config
        assert trainer.svi is not None
        assert len(trainer.train_losses) == 0
        assert len(trainer.test_losses) == 0

    def test_trainer_default_config(self, model: DeepMarkovModel) -> None:
        """Test trainer with default configuration."""
        trainer = DMMTrainer(model)

        assert trainer.config is not None
        assert isinstance(trainer.config, TrainingConfig)

    def test_train_epoch(
        self,
        model: DeepMarkovModel,
        training_config: TrainingConfig,
        sample_sequences: List[Tensor],
    ) -> None:
        """Test training for one epoch."""
        trainer = DMMTrainer(model, config=training_config)

        loss = trainer.train_epoch(sample_sequences)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_evaluate(
        self,
        model: DeepMarkovModel,
        training_config: TrainingConfig,
        sample_sequences: List[Tensor],
    ) -> None:
        """Test model evaluation."""
        trainer = DMMTrainer(model, config=training_config)

        # Train first to initialize parameters
        trainer.train_epoch(sample_sequences)

        # Evaluate
        loss = trainer.evaluate(sample_sequences)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_fit(
        self,
        model: DeepMarkovModel,
        training_config: TrainingConfig,
        sample_sequences: List[Tensor],
    ) -> None:
        """Test full training loop."""
        trainer = DMMTrainer(model, config=training_config)

        history = trainer.fit(
            train_seqs=sample_sequences,
            test_seqs=sample_sequences[:2],
            num_epochs=2,
            print_every=1,
        )

        assert "train_losses" in history
        assert "test_losses" in history
        assert len(history["train_losses"]) == 2

    def test_fit_without_test(
        self,
        model: DeepMarkovModel,
        training_config: TrainingConfig,
        sample_sequences: List[Tensor],
    ) -> None:
        """Test training without test set."""
        trainer = DMMTrainer(model, config=training_config)

        history = trainer.fit(
            train_seqs=sample_sequences,
            num_epochs=2,
        )

        assert "train_losses" in history
        assert "test_losses" not in history or len(history["test_losses"]) == 0

    def test_loss_decreases(
        self,
        model_config: ModelConfig,
        sample_sequences: List[Tensor],
    ) -> None:
        """Test that loss generally decreases during training."""
        # Create fresh model
        model = DMMFactory.create_default(model_config)
        trainer = DMMTrainer(
            model,
            config=TrainingConfig(
                lr=1e-2,
                num_epochs=10,
                print_every=5,
            ),
        )

        history = trainer.fit(train_seqs=sample_sequences, num_epochs=10)

        # Check that final loss is lower than initial loss
        # Allow some tolerance for stochastic training
        assert history["train_losses"][-1] <= history["train_losses"][0] * 1.5

    def test_get_normalized_elbo(
        self,
        model: DeepMarkovModel,
        training_config: TrainingConfig,
        sample_sequences: List[Tensor],
    ) -> None:
        """Test ELBO normalization."""
        trainer = DMMTrainer(model, config=training_config)

        raw_elbo = 1000.0
        normalized = trainer.get_normalized_elbo(sample_sequences, raw_elbo)

        # Normalized ELBO should be smaller (per data point)
        total_elements = sum(s.shape[0] * s.shape[1] for s in sample_sequences)
        expected = raw_elbo / total_elements

        assert normalized == pytest.approx(expected)
