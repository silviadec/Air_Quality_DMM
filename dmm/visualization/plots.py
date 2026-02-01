"""Visualization functions for the Deep Markov Model.

This module provides plotting functions for data exploration,
training diagnostics, and forecast visualization.
"""

from typing import List, Literal, Optional, Tuple

import calendar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from torch import Tensor


def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    datetime_col: str = "datetime",
    normalize: bool = True,
    title: str = "Time Series Plot",
    figsize: Tuple[int, int] = (15, 10),
) -> Figure:
    """Plot multiple time series variables.

    Args:
        df: DataFrame containing the time series data.
        columns: List of column names to plot.
        datetime_col: Name of the datetime column.
        normalize: Whether to z-score normalize the data for comparison.
        title: Plot title.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(
        nrows=len(columns), ncols=1, figsize=figsize, sharex=True
    )
    fig.suptitle(title, fontsize=16)

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        data = df[col].copy()
        if normalize:
            data = (data - data.mean()) / data.std()

        ax.plot(df[datetime_col], data, linewidth=1)
        ax.set_ylabel(f"{col} (norm.)" if normalize else col)
        ax.set_title(col)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    figsize: Tuple[int, int] = (7, 5),
) -> Figure:
    """Plot correlation matrix as a heatmap.

    Args:
        df: DataFrame containing the data.
        columns: List of column names to include.
        method: Correlation method ('pearson', 'kendall', or 'spearman').
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    import seaborn as sns

    corr_matrix = df[columns].corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": f"{method.capitalize()} correlation"},
        ax=ax,
    )

    ax.set_title(
        f"{method.capitalize()} Correlation Matrix", fontsize=14, pad=12
    )
    plt.tight_layout()

    return fig


def plot_nan_calendar(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    figsize: Tuple[int, int] = (13, 3),
) -> Figure:
    """Plot calendar view of days containing missing values.

    Args:
        df: DataFrame with potential NaN values.
        datetime_col: Name of the datetime column.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy[datetime_col]).dt.date

    # Group by date and check for NaN values
    daily_nan_series = df_copy.groupby("date").apply(
        lambda x: x.isna().any().any(), include_groups=False
    )
    daily_nan = daily_nan_series.reset_index()
    daily_nan.columns = ["date", "has_nan"]

    daily_nan["month"] = pd.to_datetime(daily_nan["date"]).dt.month
    daily_nan["day"] = pd.to_datetime(daily_nan["date"]).dt.day

    months = sorted(daily_nan["month"].unique())
    max_days = 31

    grid = np.full((len(months), max_days), np.nan)

    for i, m in enumerate(months):
        subset = daily_nan[daily_nan["month"] == m]
        for _, row in subset.iterrows():
            grid[i, int(row["day"]) - 1] = row["has_nan"]

    cmap = ListedColormap(["#f5f5f5", "#d95f5f"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(-0.5, max_days, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(months), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    month_names = [str(calendar.month_name[m]) for m in months]
    ax.set_yticks(range(len(months)))
    ax.set_yticklabels(month_names)

    ax.set_xticks(range(0, 31, 2))
    ax.set_xticklabels([str(i) for i in range(1, 32, 2)])

    ax.set_xlabel("Day of month")
    ax.set_ylabel("")
    ax.set_title("Daily Missing Values Overview", pad=10)

    plt.tight_layout()

    return fig


def plot_nan_context(
    df: pd.DataFrame,
    col: str,
    nan_idx: int,
    datetime_col: str = "datetime",
    hours_before: int = 6,
    hours_after: int = 6,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot context around a NaN value for visual inspection.

    Args:
        df: DataFrame containing the data.
        col: Column name to plot.
        nan_idx: Index of the NaN value.
        datetime_col: Name of the datetime column.
        hours_before: Hours to show before the NaN.
        hours_after: Hours to show after the NaN.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    start_idx = max(0, nan_idx - hours_before)
    end_idx = min(len(df) - 1, nan_idx + hours_after)
    data_window = df.iloc[start_idx : end_idx + 1]

    ax.plot(data_window[datetime_col], data_window[col], marker="o")
    ax.axvline(df.iloc[nan_idx][datetime_col], color="red", linestyle="--")
    ax.set_title(f"{col}, index {nan_idx}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel(col)

    return ax


def plot_training_loss(
    train_losses: List[float],
    test_loss: Optional[float] = None,
    smooth_window: int = 50,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot training loss over epochs.

    Args:
        train_losses: List of training losses per epoch.
        test_loss: Optional test loss to show as horizontal line.
        smooth_window: Window size for rolling average smoothing.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw training loss
    ax.plot(train_losses, alpha=0.3, label="Train ELBO (raw)")

    # Plot smoothed training loss
    if smooth_window > 1:
        train_series = pd.Series(train_losses)
        train_smooth = train_series.rolling(
            window=smooth_window, min_periods=1
        ).mean()
        ax.plot(train_smooth, label=f"Train ELBO (smoothed, window={smooth_window})")

    # Plot test loss if provided
    if test_loss is not None:
        ax.axhline(
            y=test_loss,
            color="orange",
            linestyle="--",
            label=f"Test ELBO = {test_loss:.4f}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO (per sequence)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig


def plot_forecast(
    observed: Tensor,
    forecast_mean: Tensor,
    forecast_lower: Optional[Tensor] = None,
    forecast_upper: Optional[Tensor] = None,
    feature_idx: int = 0,
    feature_name: str = "Value",
    true_future: Optional[Tensor] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """Plot forecast with uncertainty intervals.

    Args:
        observed: Observed sequence of shape (T_obs, features).
        forecast_mean: Mean forecast of shape (T_forecast, features).
        forecast_lower: Lower bound of shape (T_forecast, features).
        forecast_upper: Upper bound of shape (T_forecast, features).
        feature_idx: Index of the feature to plot.
        feature_name: Name of the feature for labeling.
        true_future: Optional true future values for comparison.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    obs_np = observed[:, feature_idx].detach().numpy()
    mean_np = forecast_mean[:, feature_idx].detach().numpy()

    T_obs = len(obs_np)
    T_forecast = len(mean_np)

    # X-axis indices
    x_obs = np.arange(T_obs)
    x_forecast = np.arange(T_obs, T_obs + T_forecast)

    # Plot observed
    ax.plot(x_obs, obs_np, "b-", label="Observed", linewidth=2)

    # Plot forecast mean
    ax.plot(x_forecast, mean_np, "r-", label="Forecast (mean)", linewidth=2)

    # Plot uncertainty interval
    if forecast_lower is not None and forecast_upper is not None:
        lower_np = forecast_lower[:, feature_idx].detach().numpy()
        upper_np = forecast_upper[:, feature_idx].detach().numpy()

        ax.fill_between(
            x_forecast,
            lower_np,
            upper_np,
            alpha=0.3,
            color="red",
            label="90% interval",
        )

    # Plot true future if available
    if true_future is not None:
        true_np = true_future[:, feature_idx].detach().numpy()
        ax.plot(
            x_forecast[: len(true_np)],
            true_np,
            "g--",
            label="True future",
            linewidth=2,
        )

    ax.axvline(x=T_obs, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time step")
    ax.set_ylabel(feature_name)
    ax.set_title(f"Forecast: {feature_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
