"""Factory for creating Deep Markov Model instances.

This module provides a factory class for convenient model instantiation
with default or custom components.
"""

from ..config import ModelConfig
from .base import BaseTransition, BaseEmitter, BaseCombiner
from .transition import GatedTransition
from .emitter import Emitter
from .combiner import Combiner
from .dmm import DeepMarkovModel


class DMMFactory:
    """Factory for creating pre-configured DMM instances.

    Provides static methods to create DeepMarkovModel instances with
    default components or custom implementations. This follows the
    Factory pattern to encapsulate object creation logic.
    """

    @staticmethod
    def create_default(config: ModelConfig) -> DeepMarkovModel:
        """Create DMM with default GatedTransition, Emitter, Combiner.

        Args:
            config: Model configuration specifying dimensions.

        Returns:
            A DeepMarkovModel instance with default components.

        Example:
            >>> config = ModelConfig(z_dim=16, output_dim=5)
            >>> model = DMMFactory.create_default(config)
        """
        transition = GatedTransition(config.z_dim, config.transition_dim)
        emitter = Emitter(config.z_dim, config.emission_dim, config.output_dim)
        combiner = Combiner(config.z_dim, config.rnn_dim)

        return DeepMarkovModel(
            transition,
            emitter,
            combiner,
            config.z_dim,
            config.rnn_dim,
            config.output_dim,
        )

    @staticmethod
    def create_custom(
        transition: BaseTransition,
        emitter: BaseEmitter,
        combiner: BaseCombiner,
        z_dim: int,
        rnn_dim: int = 64,
        output_dim: int = 5,
    ) -> DeepMarkovModel:
        """Create DMM with custom components.

        Allows injection of custom transition, emission, and combiner
        implementations that conform to the base interfaces.

        Args:
            transition: Custom transition network implementing BaseTransition.
            emitter: Custom emission network implementing BaseEmitter.
            combiner: Custom combiner network implementing BaseCombiner.
            z_dim: Dimensionality of the latent state.
            rnn_dim: Dimensionality of the RNN hidden state.
            output_dim: Dimensionality of the observed variables.

        Returns:
            A DeepMarkovModel instance with custom components.

        Example:
            >>> class MyTransition(BaseTransition):
            ...     # custom implementation
            >>> model = DMMFactory.create_custom(
            ...     MyTransition(16, 64),
            ...     Emitter(16, 32, 5),
            ...     Combiner(16, 64),
            ...     z_dim=16
            ... )
        """
        return DeepMarkovModel(
            transition, emitter, combiner, z_dim, rnn_dim, output_dim
        )
