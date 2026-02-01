"""Forecasting module for the Deep Markov Model.

This module implements the Forecaster class for performing
latent inference and future predictions.
"""

from typing import Optional

import torch
from torch import Tensor

from ..models.dmm import DeepMarkovModel


class DMMForecaster:
    """Forecaster for Deep Markov Model predictions.

    Provides methods for inferring latent states from observed
    sequences and generating future predictions by sampling
    from the learned dynamics.

    Attributes:
        model: The trained DeepMarkovModel.
    """

    def __init__(self, model: DeepMarkovModel) -> None:
        """Initialize the forecaster.

        Args:
            model: Trained DeepMarkovModel instance.
        """
        self.model = model

    def infer_latent(self, sequence: Tensor) -> Tensor:
        """Infer the final latent state from an observed sequence.

        Uses the guide (approximate posterior) to infer the latent
        state at the end of the sequence. Observations are encoded
        through the RNN to capture future context.

        Args:
            sequence: Observed sequence tensor of shape (T, features).

        Returns:
            Inferred latent state tensor of shape (z_dim,).
        """
        self.model.eval()

        with torch.no_grad():
            seq_batch = sequence.unsqueeze(0)  # (1, T, features)
            batch_size, T_max, _ = seq_batch.shape

            # Encode observations with RNN (captures future context)
            rnn_output = self.model._encode_observations(seq_batch, batch_size)

            # Initialize latent state
            z_t = self.model.get_z_0().unsqueeze(0)

            # Iterate through sequence using RNN context
            for t in range(T_max):
                h_rnn = rnn_output[:, t, :]
                loc_q, scale_q = self.model.combiner(z_t, h_rnn)
                # Use mean for deterministic inference
                z_t = loc_q

        return z_t.squeeze(0)

    def infer_latent_sample(
        self, sequence: Tensor, num_samples: int = 1
    ) -> Tensor:
        """Infer latent states with sampling from the posterior.

        Observations are encoded through the RNN to capture future context.

        Args:
            sequence: Observed sequence tensor of shape (T, features).
            num_samples: Number of posterior samples to draw.

        Returns:
            Sampled latent states of shape (num_samples, z_dim).
        """
        self.model.eval()
        samples = []

        with torch.no_grad():
            seq_batch = sequence.unsqueeze(0)
            batch_size, T_max, _ = seq_batch.shape

            # Encode observations with RNN (captures future context)
            rnn_output = self.model._encode_observations(seq_batch, batch_size)

            for _ in range(num_samples):
                z_t = self.model.get_z_0().unsqueeze(0)

                for t in range(T_max):
                    h_rnn = rnn_output[:, t, :]
                    loc_q, scale_q = self.model.combiner(z_t, h_rnn)
                    z_t = torch.distributions.Normal(loc_q, scale_q).sample()

                samples.append(z_t.squeeze(0))

        return torch.stack(samples)

    def forecast(
        self,
        z_last: Tensor,
        steps: int,
        num_samples: int = 100,
    ) -> Tensor:
        """Generate future predictions from a latent state.

        Samples future trajectories by iterating the transition
        and emission models forward in time.

        Args:
            z_last: Initial latent state of shape (z_dim,) or
                (num_init_samples, z_dim).
            steps: Number of future time steps to predict.
            num_samples: Number of sample trajectories to generate.

        Returns:
            Predicted observations of shape (num_samples, steps, features).

        Example:
            >>> z_last = forecaster.infer_latent(observed_sequence)
            >>> predictions = forecaster.forecast(z_last, steps=24)
            >>> mean_pred = predictions.mean(dim=0)
        """
        self.model.eval()
        samples_all = []

        # Handle both single and batched z_last
        if z_last.dim() == 1:
            z_last = z_last.unsqueeze(0)

        with torch.no_grad():
            for _ in range(num_samples):
                z_prev = z_last.clone()
                future_vals = []

                for step in range(steps):
                    # Transition: sample z_{t+1} | z_t
                    loc, scale = self.model.trans(z_prev)
                    z_new = torch.distributions.Normal(loc, scale).sample()

                    # Emission: sample x_{t+1} | z_{t+1}
                    x_loc, x_scale = self.model.emitter(z_new)
                    x_new = torch.distributions.Normal(x_loc, x_scale).sample()

                    future_vals.append(x_new.squeeze(0))
                    z_prev = z_new

                samples_all.append(torch.stack(future_vals))

        return torch.stack(samples_all)

    def forecast_mean(
        self,
        z_last: Tensor,
        steps: int,
    ) -> Tensor:
        """Generate deterministic mean predictions.

        Uses the mean of the transition and emission distributions
        instead of sampling for deterministic forecasts.

        Args:
            z_last: Initial latent state of shape (z_dim,).
            steps: Number of future time steps to predict.

        Returns:
            Mean predicted observations of shape (steps, features).
        """
        self.model.eval()

        if z_last.dim() == 1:
            z_last = z_last.unsqueeze(0)

        with torch.no_grad():
            z_prev = z_last
            predictions = []

            for step in range(steps):
                # Use mean of transition
                loc, _ = self.model.trans(z_prev)
                z_prev = loc

                # Use mean of emission
                x_loc, _ = self.model.emitter(z_prev)
                predictions.append(x_loc.squeeze(0))

        return torch.stack(predictions)

    def get_prediction_intervals(
        self,
        forecast_samples: Tensor,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute prediction intervals from forecast samples.

        Args:
            forecast_samples: Sampled predictions of shape
                (num_samples, steps, features).
            lower_quantile: Lower quantile for interval (default 10%).
            upper_quantile: Upper quantile for interval (default 90%).

        Returns:
            Tuple of (mean, lower_bound, upper_bound), each of
            shape (steps, features).
        """
        mean = forecast_samples.mean(dim=0)
        lower = forecast_samples.quantile(lower_quantile, dim=0)
        upper = forecast_samples.quantile(upper_quantile, dim=0)

        return mean, lower, upper
