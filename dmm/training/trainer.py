"""Training module for the Deep Markov Model.

This module implements the Trainer class that handles SVI training
with ELBO loss for the Deep Markov Model.
"""

from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
from torch import Tensor
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

from ..config import TrainingConfig
from ..models.dmm import DeepMarkovModel


class DMMTrainer:
    """Trainer class for Deep Markov Model.

    Handles stochastic variational inference (SVI) training using
    the ELBO objective. Supports training with variable-length
    sequences and provides methods for evaluation.

    Attributes:
        model: The DeepMarkovModel to train.
        config: Training configuration.
        svi: Pyro SVI object for optimization.
        train_losses: List of training losses per epoch.
        test_losses: List of test losses per epoch (if evaluated).
    """

    def __init__(
        self,
        model: DeepMarkovModel,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: DeepMarkovModel instance to train.
            config: Training configuration. Uses defaults if not provided.
        """
        self.model = model
        self.config = config or TrainingConfig()

        # Set up optimizer
        adam_params = {
            "lr": self.config.lr,
            "betas": self.config.betas,
            "clip_norm": self.config.clip_norm,
        }
        optimizer = optim.ClippedAdam(adam_params)

        # Set up SVI
        self.svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

        # Training history
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []

    def train_epoch(self, train_seqs: List[Tensor]) -> float:
        """Train for one epoch over all sequences.

        Args:
            train_seqs: List of training sequence tensors.

        Returns:
            Average ELBO loss for the epoch.
        """
        epoch_loss = 0.0

        # Shuffle sequences
        indices = list(range(len(train_seqs)))
        np.random.shuffle(indices)

        for i in indices:
            seq = train_seqs[i].unsqueeze(0)  # shape (1, T, D)
            length = torch.tensor([seq.shape[1]])

            loss = float(cast(Any, self.svi.step(seq, length)))
            epoch_loss += loss

        return epoch_loss / len(train_seqs)

    def evaluate(self, test_seqs: List[Tensor]) -> float:
        """Evaluate model on test sequences.

        Args:
            test_seqs: List of test sequence tensors.

        Returns:
            Average ELBO loss on test set.
        """
        test_loss = 0.0

        for seq in test_seqs:
            seq_batch = seq.unsqueeze(0)
            length = torch.tensor([seq_batch.shape[1]])

            loss = float(cast(Any, self.svi.evaluate_loss(seq_batch, length)))
            test_loss += loss

        return test_loss / len(test_seqs)

    def fit(
        self,
        train_seqs: List[Tensor],
        test_seqs: Optional[List[Tensor]] = None,
        num_epochs: Optional[int] = None,
        print_every: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            train_seqs: List of training sequence tensors.
            test_seqs: Optional list of test sequence tensors for evaluation.
            num_epochs: Number of epochs to train. Uses config default if None.
            print_every: Print progress every N epochs. Uses config default if None.

        Returns:
            Dictionary with 'train_losses' and optionally 'test_losses'.

        Example:
            >>> history = trainer.fit(train_seqs, test_seqs)
            >>> plt.plot(history['train_losses'])
        """
        num_epochs = num_epochs or self.config.num_epochs
        print_every = print_every or self.config.print_every

        np.random.seed(self.config.random_seed)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_seqs)
            self.train_losses.append(train_loss)

            if epoch % print_every == 0:
                msg = f"Epoch {epoch}: train ELBO = {train_loss:.4f}"

                if test_seqs is not None:
                    test_loss = self.evaluate(test_seqs)
                    self.test_losses.append(test_loss)
                    msg += f", test ELBO = {test_loss:.4f}"

                print(msg)

        result = {"train_losses": self.train_losses}
        if test_seqs is not None:
            result["test_losses"] = self.test_losses

        return result

    def get_normalized_elbo(
        self, sequences: List[Tensor], elbo: float
    ) -> float:
        """Compute ELBO normalized per data point.

        Args:
            sequences: List of sequence tensors used to compute ELBO.
            elbo: Raw ELBO value.

        Returns:
            ELBO normalized by total number of observed elements.
        """
        total_elements = sum(s.shape[0] * s.shape[1] for s in sequences)
        return elbo / total_elements
