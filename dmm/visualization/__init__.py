"""Visualization module for the Deep Markov Model.

This package provides plotting functions for data exploration,
training diagnostics, and forecast visualization.
"""

from .plots import (
    plot_time_series,
    plot_correlation_matrix,
    plot_nan_calendar,
    plot_nan_context,
    plot_training_loss,
    plot_forecast,
)

__all__ = [
    "plot_time_series",
    "plot_correlation_matrix",
    "plot_nan_calendar",
    "plot_nan_context",
    "plot_training_loss",
    "plot_forecast",
]
