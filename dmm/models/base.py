"""Abstract base classes for Deep Markov Model components.

This module defines interfaces for transition, emission, and combiner networks
following the Interface Segregation and Dependency Inversion principles.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseTransition(nn.Module, ABC):
    """Interface for transition models p(z_t | z_{t-1}).

    Transition models define the prior distribution over latent states
    given the previous latent state. Implementations should output
    parameters of a Gaussian distribution.
    """

    @abstractmethod
    def forward(self, z_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute transition distribution parameters.

        Args:
            z_prev: Previous latent state of shape (batch_size, z_dim).

        Returns:
            Tuple of (loc, scale) tensors for the Gaussian distribution,
            each of shape (batch_size, z_dim).
        """
        pass


class BaseEmitter(nn.Module, ABC):
    """Interface for emission models p(x_t | z_t).

    Emission models define the likelihood of observations given the
    current latent state. Implementations should output parameters
    of a Gaussian distribution over observations.
    """

    @abstractmethod
    def forward(self, z_t: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute emission distribution parameters.

        Args:
            z_t: Current latent state of shape (batch_size, z_dim).

        Returns:
            Tuple of (loc, scale) tensors for the Gaussian distribution,
            each of shape (batch_size, output_dim).
        """
        pass


class BaseCombiner(nn.Module, ABC):
    """Interface for combiner/guide networks q(z_t | z_{t-1}, h_rnn).

    Combiner models define the approximate posterior distribution over
    latent states given the previous latent state and RNN hidden state
    encoding future observations. Implementations should output
    parameters of a Gaussian distribution.
    """

    @abstractmethod
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
        pass
