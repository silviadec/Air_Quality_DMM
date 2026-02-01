"""Deep Markov Model implementation.

This module implements the DeepMarkovModel class that combines
transition, emission, and combiner components for probabilistic
time series modeling.
"""

import torch
import torch.nn as nn
from torch import Tensor
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from .base import BaseTransition, BaseEmitter, BaseCombiner


class DeepMarkovModel(PyroModule):
    """Deep Markov Model for probabilistic time series modeling.

    A Deep Markov Model combines a state space model structure with
    deep neural networks for the transition and emission distributions.
    The model defines:
        - Prior: p(z_t | z_{t-1}) via transition network
        - Likelihood: p(x_t | z_t) via emission network
        - Posterior: q(z_t | z_{t-1}, x_{t:T}) via combiner network

    This implementation follows the Dependency Inversion principle by
    accepting abstract base classes for all components, allowing
    different implementations to be swapped without modifying this class.

    Attributes:
        trans: Transition network for p(z_t | z_{t-1}).
        emitter: Emission network for p(x_t | z_t).
        combiner: Combiner network for q(z_t | z_{t-1}, x_{t:T}).
        z_dim: Dimensionality of the latent state.
        rnn_dim: Dimensionality of the RNN hidden state.
        output_dim: Dimensionality of the observed variables.
        rnn: GRU encoder for processing observations.
        z_0_mu: Trainable initial latent state mean (for model).
        z_q_0_mu: Trainable initial latent state mean (for guide).
        h_0: Trainable initial RNN hidden state.
    """

    def __init__(
        self,
        transition: BaseTransition,
        emitter: BaseEmitter,
        combiner: BaseCombiner,
        z_dim: int,
        rnn_dim: int = 64,
        output_dim: int = 5,
    ) -> None:
        """Initialize the Deep Markov Model.

        Args:
            transition: Transition network implementing BaseTransition.
            emitter: Emission network implementing BaseEmitter.
            combiner: Combiner network implementing BaseCombiner.
            z_dim: Dimensionality of the latent state.
            rnn_dim: Dimensionality of the RNN hidden state.
            output_dim: Dimensionality of the observed variables.
        """
        super().__init__()

        self.trans = transition
        self.emitter = emitter
        self.combiner = combiner
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        self.output_dim = output_dim

        # RNN encoder for processing observations (captures future context)
        self.rnn = nn.GRU(
            input_size=output_dim,
            hidden_size=rnn_dim,
            batch_first=True,
            bidirectional=False,
        )

        # Trainable initial latent state parameters (for model/prior)
        self.z_0_mu = nn.Parameter(torch.zeros(z_dim))
        self.z_0_sigma = nn.Parameter(torch.ones(z_dim))

        # Trainable initial latent state parameters (for guide/posterior)
        self.z_q_0_mu = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0_sigma = nn.Parameter(torch.ones(z_dim))

        # Trainable initial RNN hidden state
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

    def _get_initial_z_model(self, batch_size: int) -> Tensor:
        """Get the initial latent state mean for the model (prior).

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            Initial latent state mean tensor of shape (batch_size, z_dim).
        """
        return self.z_0_mu.expand(batch_size, -1)

    def _get_initial_z_guide(self, batch_size: int) -> Tensor:
        """Get the initial latent state mean for the guide (posterior).

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            Initial latent state mean tensor of shape (batch_size, z_dim).
        """
        return self.z_q_0_mu.expand(batch_size, -1)

    def _create_mask(
        self, batch_size: int, T_max: int, lengths: Tensor, device: torch.device
    ) -> Tensor:
        """Create a mask tensor for variable-length sequences.

        Args:
            batch_size: Number of sequences in the batch.
            T_max: Maximum sequence length.
            lengths: Actual lengths of each sequence.
            device: Device to create tensor on.

        Returns:
            Mask tensor of shape (batch_size, T_max) with 1s for valid timesteps.
        """
        mask = torch.zeros(batch_size, T_max, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0
        return mask

    def _encode_observations(
        self, sequences: Tensor, batch_size: int
    ) -> Tensor:
        """Encode observations using the RNN in reverse order.

        Processes sequences in reverse to capture future context at each timestep.

        Args:
            sequences: Observation sequences of shape (batch, T_max, features).
            batch_size: Number of sequences in the batch.

        Returns:
            RNN hidden states of shape (batch, T_max, rnn_dim).
        """
        # Reverse sequences to capture future context
        seq_reversed = torch.flip(sequences, [1])

        # Initialize RNN hidden state
        h_0 = self.h_0.expand(1, batch_size, -1).contiguous()

        # Process through RNN
        rnn_output, _ = self.rnn(seq_reversed, h_0)

        # Reverse back to original order
        rnn_output = torch.flip(rnn_output, [1])

        return rnn_output

    def model(self, sequences: Tensor, lengths: Tensor) -> None:
        """Generative model p(x_{1:T}, z_{1:T}).

        Defines the joint distribution over observations and latent
        states by iterating through time steps, sampling latent states
        from the transition prior and observations from the emission
        likelihood.

        Args:
            sequences: Observation sequences of shape (batch, T_max, features).
            lengths: Actual lengths of each sequence in the batch.
        """
        pyro.module("dmm", self)

        batch_size, T_max, _ = sequences.shape
        z_prev = self._get_initial_z_model(batch_size)

        # Create mask for variable-length sequences
        mask = self._create_mask(batch_size, T_max, lengths, sequences.device)

        with pyro.plate("batch", batch_size):
            for t in range(T_max):
                # Transition: p(z_t | z_{t-1})
                loc, scale = self.trans(z_prev)

                # Apply mask to handle variable-length sequences
                z_t = pyro.sample(
                    f"z_{t}",
                    dist.Normal(loc, scale)
                    .mask(mask[:, t : t + 1])
                    .to_event(1),
                )

                # Emission: p(x_t | z_t)
                x_loc, x_scale = self.emitter(z_t)
                pyro.sample(
                    f"x_{t}",
                    dist.Normal(x_loc, x_scale)
                    .mask(mask[:, t : t + 1])
                    .to_event(1),
                    obs=sequences[:, t, :],
                )

                z_prev = z_t

    def guide(self, sequences: Tensor, lengths: Tensor) -> None:
        """Variational guide q(z_{1:T} | x_{1:T}).

        Defines the approximate posterior over latent states using the
        combiner network. The guide processes observations through an RNN
        (in reverse order to capture future context) and combines them
        with the previous latent state to parameterize the approximate
        posterior at each time step.

        Args:
            sequences: Observation sequences of shape (batch, T_max, features).
            lengths: Actual lengths of each sequence in the batch.
        """
        pyro.module("dmm", self)

        batch_size, T_max, _ = sequences.shape

        # Encode observations with RNN (captures future context)
        rnn_output = self._encode_observations(sequences, batch_size)

        # Create mask for variable-length sequences
        mask = self._create_mask(batch_size, T_max, lengths, sequences.device)

        z_prev = self._get_initial_z_guide(batch_size)

        with pyro.plate("batch", batch_size):
            for t in range(T_max):
                # Get RNN hidden state at this timestep
                h_rnn = rnn_output[:, t, :]

                # Approximate posterior: q(z_t | z_{t-1}, h_rnn)
                loc_q, scale_q = self.combiner(z_prev, h_rnn)

                # Apply mask to handle variable-length sequences
                z_t = pyro.sample(
                    f"z_{t}",
                    dist.Normal(loc_q, scale_q)
                    .mask(mask[:, t : t + 1])
                    .to_event(1),
                )

                z_prev = z_t

    def get_z_0(self) -> Tensor:
        """Get the initial latent state mean (for inference/forecasting).

        Returns the guide's initial latent state mean, which is trainable.

        Returns:
            Initial latent state mean tensor of shape (z_dim,).
        """
        return self.z_q_0_mu
