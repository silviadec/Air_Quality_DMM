"""Data preprocessing functions for the Deep Markov Model.

This module provides functions for cleaning, normalizing, and
preparing time series data for model training.
"""

from typing import Any, List, Tuple, cast

import numpy as np
import pandas as pd


def zscore_normalize(
    series: pd.Series,  # type: ignore[type-arg]
) -> Tuple[pd.Series, float, float]:  # type: ignore[type-arg]
    """Apply z-score normalization to a pandas Series.

    Transforms data to have zero mean and unit standard deviation.

    Args:
        series: Input pandas Series to normalize.

    Returns:
        Tuple of (normalized_series, mean, std) where:
            - normalized_series: Series with mean=0 and std=1
            - mean: Original mean value
            - std: Original standard deviation

    Example:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> normalized, mean, std = zscore_normalize(s)
    """
    mean = float(series.mean())
    std = float(series.std())
    if std == 0:
        return series - mean, mean, 1.0
    return (series - mean) / std, mean, std


def fill_short_nan_intervals(
    df: pd.DataFrame, col: str, max_gap: int = 2
) -> pd.DataFrame:
    """Fill short NaN intervals with linear interpolation.

    Identifies consecutive NaN sequences in the specified column
    and fills those with length <= max_gap using linear interpolation
    between the surrounding valid values.

    Args:
        df: DataFrame containing the column to process.
        col: Column name to process.
        max_gap: Maximum consecutive NaN gap to fill. Gaps longer than
            this are left as NaN.

    Returns:
        DataFrame with short NaN intervals filled.

    Example:
        >>> df = pd.DataFrame({'val': [1, np.nan, 3, np.nan, np.nan, np.nan, 7]})
        >>> df = fill_short_nan_intervals(df, 'val', max_gap=2)
        >>> # Only the first NaN is filled (gap=1), longer gap remains
    """
    df = df.copy()
    idx = 0

    while idx < len(df):
        if pd.isna(df.loc[idx, col]):
            start = idx
            end = idx

            # Find the end of the NaN sequence
            while end + 1 < len(df) and pd.isna(df.loc[end + 1, col]):
                end += 1

            gap_length = end - start + 1

            # Only fill if gap is short enough and has valid neighbors
            if gap_length <= max_gap:
                if start > 0 and end < len(df) - 1:
                    prev_val = df.loc[start - 1, col]
                    next_val = df.loc[end + 1, col]

                    if not pd.isna(prev_val) and not pd.isna(next_val):
                        # Linear interpolation
                        prev_float = float(cast(Any, prev_val))
                        next_float = float(cast(Any, next_val))
                        for i, pos in enumerate(range(start, end + 1)):
                            df.loc[pos, col] = prev_float + (next_float - prev_float) * (
                                i + 1
                            ) / (gap_length + 1)

            idx = end + 1
        else:
            idx += 1

    return df


class NormalizationStats:
    """Container for normalization statistics.

    Stores mean and std for each normalized column to enable
    inverse transformation of predictions.

    Attributes:
        stats: Dictionary mapping column names to (mean, std) tuples.
    """

    def __init__(self) -> None:
        self.stats: dict[str, Tuple[float, float]] = {}

    def add(self, column: str, mean: float, std: float) -> None:
        """Add normalization stats for a column."""
        self.stats[column] = (mean, std)

    def get(self, column: str) -> Tuple[float, float]:
        """Get (mean, std) for a column."""
        return self.stats[column]

    def denormalize(self, values: np.ndarray, column: str) -> np.ndarray:
        """Convert normalized values back to original scale.

        Args:
            values: Normalized values (numpy array).
            column: Column name to get stats for.

        Returns:
            Values in original scale.
        """
        mean, std = self.stats[column]
        return values * std + mean

    def denormalize_all(
        self, values: np.ndarray, columns: Tuple[str, ...]
    ) -> np.ndarray:
        """Denormalize all columns in a 2D array.

        Args:
            values: Array of shape (T, num_columns) with normalized values.
            columns: Tuple of column names in order.

        Returns:
            Array with values in original scale.
        """
        result = np.zeros_like(values)
        for i, col in enumerate(columns):
            result[:, i] = self.denormalize(values[:, i], col)
        return result


def preprocess_dataframe(
    df: pd.DataFrame,
    columns: Tuple[str, ...],
    max_nan_gap: int = 2,
) -> Tuple[pd.DataFrame, NormalizationStats]:
    """Preprocess a DataFrame for model training.

    Applies z-score normalization to specified columns and fills
    short NaN intervals with linear interpolation.

    Args:
        df: Input DataFrame with raw data.
        columns: Tuple of column names to process.
        max_nan_gap: Maximum consecutive NaN gap to fill.

    Returns:
        Tuple of (preprocessed_dataframe, normalization_stats) where:
            - preprocessed_dataframe: DataFrame with normalized values
            - normalization_stats: NormalizationStats object for inverse transform

    Example:
        >>> df, stats = preprocess_dataframe(df, ('PM10', 'temp', 'humidity'))
        >>> original_values = stats.denormalize(predictions, 'PM10')
    """
    df_processed = df.copy()
    norm_stats = NormalizationStats()

    # Apply z-score normalization and store stats
    for col in columns:
        if col in df_processed.columns:
            normalized, mean, std = zscore_normalize(df_processed[col])
            df_processed[col] = normalized
            norm_stats.add(col, mean, std)

    # Fill short NaN intervals
    for col in columns:
        if col in df_processed.columns:
            df_processed = fill_short_nan_intervals(df_processed, col, max_nan_gap)

    return df_processed, norm_stats


def nan_summary(df: pd.DataFrame, time_col: str = "datetime") -> Tuple[List[str], int]:
    """Generate a summary of missing values in the DataFrame.

    Prints statistics about NaN values including total counts,
    per-column counts, and consecutive NaN interval information.

    Args:
        df: DataFrame to analyze.
        time_col: Name of the datetime column.

    Returns:
        Tuple of (columns_with_nan, total_nan_rows).
    """
    cols_with_nan = df.columns[df.isna().any()].tolist()
    nan_row_count = int(df.isna().any(axis=1).sum())
    perc_nan_rows = nan_row_count / len(df) * 100

    print(f"Total number of rows: {len(df)}")
    print(f"Rows with at least one NaN: {nan_row_count} ({perc_nan_rows:.2f}%)")
    print(f"Columns with NaNs: {cols_with_nan}")

    print("\nNaN count per column:")
    print(df[cols_with_nan].isna().sum())

    # Identify consecutive NaN sequences per column
    print("\nConsecutive NaN intervals per column:")
    for col in cols_with_nan:
        is_nan = df[col].isna()
        group_id = (is_nan != is_nan.shift()).cumsum()
        nan_groups: pd.Series[int] = is_nan.groupby(group_id).sum()  # type: ignore[type-arg]
        lengths = nan_groups[cast(Any, nan_groups.values) > 0]

        if not lengths.empty:
            print(f"{col}: {len(lengths)} intervals, lengths = {list(lengths)}")
        else:
            print(f"{col}: no consecutive intervals")

    return cols_with_nan, nan_row_count
