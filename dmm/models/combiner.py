"""Combiner model for Deep Markov Model guide.

This module implements the combiner network that parameterizes
the approximate posterior q(z_t | z_{t-1}, x_{t:T}).
"""

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .base import BaseCombiner


class Combiner(BaseCombiner):
    """Combiner network for approximate posterior inference.

    Combines information from the previous latent state and an RNN
    hidden state (encoding future observations) to parameterize
    the approximate posterior distribution over the current latent state.

    The combination is done via element-wise addition after
    projecting the previous latent state to the RNN hidden dimension:
        h_combined = tanh(W * z_{t-1}) + h_rnn
        q(z_t | z_{t-1}, x_{t:T}) = Normal(loc(h_combined), scale(h_combined))

    Attributes:
        lin_z_to_hidden: Projects previous latent to hidden dimension.
        lin_hidden_to_loc: Output layer for location parameter.
        lin_hidden_to_scale: Output layer for scale parameter.
    """

    def __init__(self, z_dim: int, rnn_dim: int) -> None:
        """Initialize the combiner network.

        Args:
            z_dim: Dimensionality of the latent state.
            rnn_dim: Dimensionality of the RNN hidden state.
        """
        super().__init__()

        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_prev: Tensor, h_rnn: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute approximate posterior distribution parameters.

        Args:
            z_prev: Previous latent state of shape (batch_size, z_dim).
            h_rnn: RNN hidden state encoding observations,
                shape (batch_size, rnn_dim).

        Returns:
            Tuple of (loc, scale) tensors for the Gaussian distribution,
            each of shape (batch_size, z_dim).
        """
        # Project z_prev to RNN hidden dimension and combine with h_rnn
        h_z = self.tanh(self.lin_z_to_hidden(z_prev))
        h_combined = h_z + h_rnn

        # Compute distribution parameters
        loc = self.lin_hidden_to_loc(h_combined)
        # Add small epsilon for numerical stability
        scale = self.softplus(self.lin_hidden_to_scale(h_combined)) + 1e-6

        return loc, scale
