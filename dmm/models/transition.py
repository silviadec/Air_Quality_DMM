"""Gated transition model for Deep Markov Model.

This module implements a gated transition network that models
the prior distribution p(z_t | z_{t-1}) over latent states.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseTransition


class GatedTransition(BaseTransition):
    """Gated transition network for latent state dynamics.

    Implements a gated transition mechanism that combines a linear
    transformation with a nonlinear proposed mean, controlled by
    a learned gating mechanism. This allows the model to interpolate
    between linear and nonlinear dynamics.

    The transition is defined as:
        gate = sigmoid(MLP(z_{t-1}))
        proposed_mean = MLP(z_{t-1})
        loc = (1 - gate) * linear(z_{t-1}) + gate * proposed_mean
        scale = softplus(MLP(proposed_mean))

    Attributes:
        lin_gate_z_to_hidden: Linear layer for gate computation.
        lin_gate_hidden_to_z: Output layer for gate.
        lin_prop_mean_z_to_hidden: Linear layer for proposed mean.
        lin_prop_mean_hidden_to_z: Output layer for proposed mean.
        lin_z_to_loc: Linear transformation (initialized to identity).
        lin_sig: Layer for computing scale parameter.
    """

    def __init__(self, z_dim: int, transition_dim: int) -> None:
        """Initialize the gated transition network.

        Args:
            z_dim: Dimensionality of the latent state.
            transition_dim: Hidden dimension for the transition network.
        """
        super().__init__()

        # Gate network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)

        # Proposed mean network
        self.lin_prop_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_prop_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)

        # Linear transformation (initialized to identity)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        # Scale network
        self.lin_sig = nn.Linear(z_dim, z_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute transition distribution parameters.

        Args:
            z_prev: Previous latent state of shape (batch_size, z_dim).

        Returns:
            Tuple of (loc, scale) tensors for the Gaussian distribution,
            each of shape (batch_size, z_dim).
        """
        # Compute gate
        gate_hidden = self.relu(self.lin_gate_z_to_hidden(z_prev))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(gate_hidden))

        # Compute proposed mean
        prop_hidden = self.relu(self.lin_prop_mean_z_to_hidden(z_prev))
        prop_mean = self.lin_prop_mean_hidden_to_z(prop_hidden)

        # Combine linear and nonlinear components
        loc = (1 - gate) * self.lin_z_to_loc(z_prev) + gate * prop_mean

        # Compute scale (always positive via softplus + epsilon for numerical stability)
        scale = self.softplus(self.lin_sig(self.relu(prop_mean))) + 1e-6

        return loc, scale
