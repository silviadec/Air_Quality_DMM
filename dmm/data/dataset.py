"""Dataset classes and sequence building utilities.

This module provides functions for splitting data into contiguous
sequences and a PyTorch Dataset wrapper for training.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


def build_sequences(
    df: pd.DataFrame,
    min_length: int = 1,
) -> List[pd.DataFrame]:
    """Split DataFrame into contiguous sequences without NaN values.

    Identifies rows with any NaN values and splits the DataFrame
    at those points, returning a list of contiguous sequences
    containing only valid data.

    Args:
        df: DataFrame with potential NaN values.
        min_length: Minimum sequence length to include. Sequences shorter
            than this will be filtered out. Default is 1 (no filtering).

    Returns:
        List of DataFrame segments, each containing contiguous
        observations without NaN values and at least min_length rows.

    Example:
        >>> sequences = build_sequences(df_preprocessed, min_length=24)
        >>> print(f"Found {len(sequences)} contiguous sequences")
    """
    is_nan = df.isna().any(axis=1)
    group_id = (is_nan != is_nan.shift()).cumsum()

    sequences = [
        group.reset_index(drop=True)
        for _, group in df.groupby(group_id)
        if not group.isna().any().any() and len(group) >= min_length
    ]

    return sequences


def sequences_to_tensors(sequences: List[pd.DataFrame]) -> List[Tensor]:
    """Convert list of DataFrame sequences to PyTorch tensors.

    Args:
        sequences: List of DataFrame sequences.

    Returns:
        List of float32 tensors, each of shape (T, features).

    Example:
        >>> tensors = sequences_to_tensors(sequences)
        >>> print(tensors[0].shape)  # (T, features)
    """
    return [
        torch.tensor(seq.to_numpy(), dtype=torch.float32)
        for seq in sequences
    ]


def train_test_split_sequences(
    sequences: List[Tensor],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Split sequences into training and test sets.

    Args:
        sequences: List of sequence tensors.
        test_size: Fraction of sequences for test set.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_sequences, test_sequences).

    Example:
        >>> train_seqs, test_seqs = train_test_split_sequences(tensors)
    """
    np.random.seed(random_seed)
    n_test = int(len(sequences) * test_size)
    indices = np.random.permutation(len(sequences))

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_seqs = [sequences[i] for i in train_indices]
    test_seqs = [sequences[i] for i in test_indices]

    return train_seqs, test_seqs


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapper for sequence data.

    Wraps a list of sequence tensors for use with PyTorch DataLoader.
    Each item is a tuple of (sequence, length) where sequence has
    shape (T, features) and length is a scalar tensor.

    Attributes:
        sequences: List of sequence tensors.
    """

    def __init__(self, sequences: List[Tensor]) -> None:
        """Initialize the dataset.

        Args:
            sequences: List of sequence tensors, each of shape (T, features).
        """
        self.sequences = sequences

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a sequence and its length.

        Args:
            idx: Index of the sequence.

        Returns:
            Tuple of (sequence, length) where sequence has shape
            (T, features) and length is a scalar tensor.
        """
        seq = self.sequences[idx]
        length = torch.tensor(seq.shape[0])
        return seq, length
