"""Emission model for Deep Markov Model.

This module implements the emission network that models
the observation likelihood p(x_t | z_t).
"""

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .base import BaseEmitter


class Emitter(BaseEmitter):
    """Emission network for observation generation.

    Maps latent states to observation distribution parameters.
    The emission distribution is a diagonal Gaussian:
        p(x_t | z_t) = Normal(loc(z_t), scale(z_t))

    Attributes:
        lin_z_to_hidden: Linear layer mapping latent to hidden.
        lin_hidden_to_loc: Output layer for location parameter.
        lin_hidden_to_scale: Output layer for scale parameter.
    """

    def __init__(self, z_dim: int, emission_dim: int, output_dim: int) -> None:
        """Initialize the emission network.

        Args:
            z_dim: Dimensionality of the latent state.
            emission_dim: Hidden dimension for the emission network.
            output_dim: Dimensionality of the observations.
        """
        super().__init__()

        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, output_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute emission distribution parameters.

        Args:
            z_t: Current latent state of shape (batch_size, z_dim).

        Returns:
            Tuple of (loc, scale) tensors for the Gaussian distribution,
            each of shape (batch_size, output_dim).
        """
        hidden = self.relu(self.lin_z_to_hidden(z_t))
        loc = self.lin_hidden_to_loc(hidden)
        # Add small epsilon for numerical stability
        scale = self.softplus(self.lin_hidden_to_scale(hidden)) + 1e-6

        return loc, scale
