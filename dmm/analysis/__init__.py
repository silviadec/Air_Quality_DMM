"""Analysis and interpretability tools for the Deep Markov Model.

This package provides tools to understand the hidden dynamics
learned by the DMM, including latent visualization, regime detection,
and anomaly analysis.
"""

from .interpreter import DMMInterpreter

__all__ = ["DMMInterpreter"]
