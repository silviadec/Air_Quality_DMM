"""Interpretability tools for Deep Markov Model analysis.

This module provides tools to understand the hidden dynamics learned
by the DMM, including latent space visualization, regime detection,
feature-latent correlation analysis, and anomaly detection.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from ..models.dmm import DeepMarkovModel
from ..inference.forecaster import DMMForecaster


class DMMInterpreter:
    """Interpretability tools for analyzing DMM hidden dynamics.

    Provides methods to visualize and understand the latent states
    learned by the Deep Markov Model, including:
    - Latent trajectory visualization (PCA/t-SNE)
    - Pollution regime detection via clustering
    - Feature-latent correlation analysis
    - Anomaly detection based on reconstruction error

    Attributes:
        model: The trained DeepMarkovModel.
        forecaster: DMMForecaster for latent inference.
    """

    def __init__(
        self,
        model: DeepMarkovModel,
        forecaster: Optional[DMMForecaster] = None,
    ) -> None:
        """Initialize the interpreter.

        Args:
            model: Trained DeepMarkovModel instance.
            forecaster: Optional DMMForecaster. Created if not provided.
        """
        self.model = model
        self.forecaster = forecaster or DMMForecaster(model)

    def infer_latent_trajectory(
        self,
        sequence: Tensor,
    ) -> Tensor:
        """Infer the full latent trajectory for a sequence.

        Args:
            sequence: Observation sequence of shape (T, features).

        Returns:
            Latent trajectory of shape (T, z_dim).
        """
        self.model.eval()
        with torch.no_grad():
            seq_batch = sequence.unsqueeze(0)
            batch_size, T_max, _ = seq_batch.shape

            # Encode observations with RNN
            rnn_output = self.model._encode_observations(seq_batch, batch_size)

            # Initialize latent state
            z_t = self.model.get_z_0().unsqueeze(0)

            # Collect all latent states
            latent_trajectory = []

            for t in range(T_max):
                h_rnn = rnn_output[:, t, :]
                loc_q, _scale_q = self.model.combiner(z_t, h_rnn)
                z_t = loc_q  # Use mean for deterministic trajectory
                latent_trajectory.append(z_t.squeeze(0))

        return torch.stack(latent_trajectory)

    def infer_all_trajectories(
        self,
        sequences: List[Tensor],
    ) -> Tuple[Tensor, List[int]]:
        """Infer latent trajectories for multiple sequences.

        Args:
            sequences: List of observation sequences.

        Returns:
            Tuple of:
                - Concatenated latent states of shape (total_T, z_dim)
                - List of sequence lengths for indexing
        """
        all_latents = []
        lengths = []

        for seq in sequences:
            trajectory = self.infer_latent_trajectory(seq)
            all_latents.append(trajectory)
            lengths.append(len(trajectory))

        return torch.cat(all_latents, dim=0), lengths

    def reduce_dimensions(
        self,
        latents: Tensor,
        method: str = "pca",
        n_components: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """Reduce latent dimensions for visualization.

        Args:
            latents: Latent states of shape (N, z_dim).
            method: Dimensionality reduction method ('pca' or 'tsne').
            n_components: Number of output dimensions.
            **kwargs: Additional arguments for the reduction method.

        Returns:
            Reduced latents of shape (N, n_components).
        """
        latents_np = latents.numpy()

        if method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=kwargs.get("perplexity", 30),
                random_state=kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

        return reducer.fit_transform(latents_np)

    def plot_latent_trajectory(
        self,
        sequence: Tensor,
        method: str = "pca",
        timestamps: Optional[pd.DatetimeIndex] = None,
        feature_name: Optional[str] = None,
        feature_values: Optional[Tensor] = None,
        figsize: Tuple[int, int] = (14, 5),
    ) -> Figure:
        """Visualize latent space evolution over time.

        Creates a plot showing how latent states evolve, optionally
        colored by a feature value (e.g., PM10 levels).

        Args:
            sequence: Observation sequence of shape (T, features).
            method: Dimensionality reduction method ('pca' or 'tsne').
            timestamps: Optional datetime index for x-axis.
            feature_name: Name of feature for coloring.
            feature_values: Feature values for coloring points.
            figsize: Figure size.

        Returns:
            Matplotlib figure with the trajectory plot.
        """
        # Infer latent trajectory
        trajectory = self.infer_latent_trajectory(sequence)
        reduced = self.reduce_dimensions(trajectory, method=method)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: 2D latent space trajectory
        ax1 = axes[0]
        if feature_values is not None:
            scatter = ax1.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=feature_values.numpy(),
                cmap="RdYlGn_r",
                alpha=0.7,
                s=20,
            )
            plt.colorbar(scatter, ax=ax1, label=feature_name or "Feature")
        else:
            # Color by time
            colors = np.arange(len(reduced))
            scatter = ax1.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=colors,
                cmap="viridis",
                alpha=0.7,
                s=20,
            )
            plt.colorbar(scatter, ax=ax1, label="Time step")

        # Draw trajectory lines
        ax1.plot(reduced[:, 0], reduced[:, 1], "k-", alpha=0.2, linewidth=0.5)
        ax1.set_xlabel(f"{method.upper()} 1")
        ax1.set_ylabel(f"{method.upper()} 2")
        ax1.set_title(f"Latent Space Trajectory ({method.upper()})")

        # Plot 2: Latent dimensions over time
        ax2 = axes[1]
        T = len(trajectory)
        x_axis = timestamps if timestamps is not None else np.arange(T)

        # Plot first 3 latent dimensions
        n_dims_to_plot = min(3, trajectory.shape[1])
        for i in range(n_dims_to_plot):
            ax2.plot(x_axis, trajectory[:, i].numpy(), label=f"z_{i}", alpha=0.8)

        ax2.set_xlabel("Time" if timestamps is None else "Date")
        ax2.set_ylabel("Latent value")
        ax2.set_title("Latent Dimensions Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if timestamps is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def detect_regimes(
        self,
        sequences: List[Tensor],
        n_regimes: int = 3,
        method: str = "kmeans",
    ) -> Dict[str, Any]:
        """Detect pollution regimes via clustering of latent states.

        Clusters the latent states to identify distinct "regimes"
        (e.g., high pollution, low pollution, transitional states).

        Args:
            sequences: List of observation sequences.
            n_regimes: Number of regimes to detect.
            method: Clustering method ('kmeans').

        Returns:
            Dictionary with:
                - 'labels': Regime labels for each timestep
                - 'model': Fitted clustering model
                - 'latents': All latent states
                - 'lengths': Sequence lengths for indexing
                - 'centers': Cluster centers in latent space
        """
        # Get all latent states
        all_latents, lengths = self.infer_all_trajectories(sequences)
        latents_np = all_latents.numpy()

        # Standardize for clustering
        scaler = StandardScaler()
        latents_scaled = scaler.fit_transform(latents_np)

        # Cluster
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        else:
            raise ValueError(f"Unknown method: {method}")

        labels = clusterer.fit_predict(latents_scaled)

        # Get cluster centers in original space
        centers = scaler.inverse_transform(clusterer.cluster_centers_)

        return {
            "labels": labels,
            "model": clusterer,
            "latents": all_latents,
            "lengths": lengths,
            "centers": centers,
            "scaler": scaler,
        }

    def plot_regime_characteristics(
        self,
        sequences: List[Tensor],
        regime_result: Dict,
        feature_names: Tuple[str, ...],
        figsize: Tuple[int, int] = (14, 8),
    ) -> Figure:
        """Visualize characteristics of each detected regime.

        Shows the average feature values for each regime to help
        interpret what each regime represents.

        Args:
            sequences: List of observation sequences.
            regime_result: Output from detect_regimes().
            feature_names: Names of the features.
            figsize: Figure size.

        Returns:
            Matplotlib figure with regime characteristics.
        """
        labels = regime_result["labels"]
        n_regimes = len(np.unique(labels))

        # Concatenate all observations
        all_obs = torch.cat(sequences, dim=0).numpy()

        # Compute mean and std for each regime
        regime_stats = []
        for regime in range(n_regimes):
            mask = labels == regime
            regime_obs = all_obs[mask]
            regime_stats.append({
                "mean": regime_obs.mean(axis=0),
                "std": regime_obs.std(axis=0),
                "count": mask.sum(),
            })

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Bar chart of mean values per regime
        ax1 = axes[0, 0]
        x = np.arange(len(feature_names))
        width = 0.8 / n_regimes
        cmap = plt.get_cmap("Set2")
        colors = cmap(np.linspace(0, 1, n_regimes))

        for i, stats in enumerate(regime_stats):
            offset = (i - n_regimes / 2 + 0.5) * width
            ax1.bar(
                x + offset,
                stats["mean"],
                width,
                label=f"Regime {i} (n={stats['count']})",
                color=colors[i],
                yerr=stats["std"] / np.sqrt(stats["count"]),
                capsize=3,
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, rotation=45, ha="right")
        ax1.set_ylabel("Mean value (normalized)")
        ax1.set_title("Feature Means by Regime")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Regime distribution pie chart
        ax2 = axes[0, 1]
        counts = [stats["count"] for stats in regime_stats]
        ax2.pie(
            counts,
            labels=[f"Regime {i}" for i in range(n_regimes)],
            autopct="%1.1f%%",
            colors=colors,
        )
        ax2.set_title("Regime Distribution")

        # Plot 3: Latent space with regime colors
        ax3 = axes[1, 0]
        latents = regime_result["latents"]
        reduced = self.reduce_dimensions(latents, method="pca")

        for i in range(n_regimes):
            mask = labels == i
            ax3.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[i]],
                label=f"Regime {i}",
                alpha=0.5,
                s=10,
            )

        # Plot cluster centers
        centers_reduced = self.reduce_dimensions(
            torch.tensor(regime_result["centers"], dtype=torch.float32),
            method="pca",
        )
        ax3.scatter(
            centers_reduced[:, 0],
            centers_reduced[:, 1],
            c="black",
            marker="x",
            s=200,
            linewidths=3,
            label="Centers",
        )

        ax3.set_xlabel("PCA 1")
        ax3.set_ylabel("PCA 2")
        ax3.set_title("Latent Space by Regime")
        ax3.legend()

        # Plot 4: Heatmap of feature values per regime
        ax4 = axes[1, 1]
        heatmap_data = np.array([stats["mean"] for stats in regime_stats])
        im = ax4.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")

        ax4.set_xticks(np.arange(len(feature_names)))
        ax4.set_yticks(np.arange(n_regimes))
        ax4.set_xticklabels(feature_names, rotation=45, ha="right")
        ax4.set_yticklabels([f"Regime {i}" for i in range(n_regimes)])

        # Add text annotations
        for i in range(n_regimes):
            for j in range(len(feature_names)):
                ax4.text(
                    j, i, f"{heatmap_data[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                )

        ax4.set_title("Feature Values Heatmap")
        plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        return fig

    def compute_feature_latent_correlation(
        self,
        sequences: List[Tensor],
    ) -> pd.DataFrame:
        """Compute correlation between features and latent dimensions.

        Helps understand what each latent dimension represents by
        showing which observed features it correlates with.

        Args:
            sequences: List of observation sequences.

        Returns:
            DataFrame with correlation coefficients (features x latent dims).
        """
        # Get all latent states
        all_latents, _ = self.infer_all_trajectories(sequences)
        all_obs = torch.cat(sequences, dim=0)

        latents_np = all_latents.numpy()
        obs_np = all_obs.numpy()

        # Compute correlation matrix
        n_features = obs_np.shape[1]
        n_latent = latents_np.shape[1]

        correlations = np.zeros((n_features, n_latent))

        for i in range(n_features):
            for j in range(n_latent):
                correlations[i, j] = np.corrcoef(obs_np[:, i], latents_np[:, j])[0, 1]

        return pd.DataFrame(
            correlations,
            columns=[f"z_{i}" for i in range(n_latent)],
        )

    def plot_feature_latent_correlation(
        self,
        sequences: List[Tensor],
        feature_names: Tuple[str, ...],
        figsize: Tuple[int, int] = (12, 6),
    ) -> Figure:
        """Visualize correlation between features and latent dimensions.

        Args:
            sequences: List of observation sequences.
            feature_names: Names of the features.
            figsize: Figure size.

        Returns:
            Matplotlib figure with correlation heatmap.
        """
        corr_df = self.compute_feature_latent_correlation(sequences)
        corr_df.index = feature_names

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(np.arange(corr_df.shape[1]))
        ax.set_yticks(np.arange(corr_df.shape[0]))
        ax.set_xticklabels(corr_df.columns)
        ax.set_yticklabels(corr_df.index)

        # Add correlation values as text
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                val = corr_df.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

        ax.set_title("Feature-Latent Correlation Matrix")
        ax.set_xlabel("Latent Dimensions")
        ax.set_ylabel("Observed Features")
        plt.colorbar(im, ax=ax, label="Correlation")

        plt.tight_layout()
        return fig

    def compute_reconstruction_error(
        self,
        sequence: Tensor,
    ) -> Tensor:
        """Compute reconstruction error at each timestep.

        The reconstruction error indicates how well the model
        captures the observed data at each point.

        Args:
            sequence: Observation sequence of shape (T, features).

        Returns:
            Reconstruction error of shape (T,).
        """
        self.model.eval()
        with torch.no_grad():
            trajectory = self.infer_latent_trajectory(sequence)

            errors = []
            for t in range(len(trajectory)):
                z_t = trajectory[t].unsqueeze(0)
                x_loc, x_scale = self.model.emitter(z_t)

                # Squared Mahalanobis distance (normalized by scale)
                diff = sequence[t] - x_loc.squeeze(0)
                error = (diff / x_scale.squeeze(0)).pow(2).sum()
                errors.append(error)

        return torch.stack(errors)

    def detect_anomalies(
        self,
        sequence: Tensor,
        threshold: float = 2.0,
        method: str = "reconstruction",
    ) -> Dict[str, Any]:
        """Detect anomalous periods based on reconstruction error.

        Identifies timesteps where the model poorly reconstructs
        the observations, indicating unusual patterns.

        Args:
            sequence: Observation sequence of shape (T, features).
            threshold: Number of standard deviations for anomaly threshold.
            method: Detection method ('reconstruction' or 'latent_velocity').

        Returns:
            Dictionary with:
                - 'scores': Anomaly scores for each timestep
                - 'threshold': Computed threshold value
                - 'anomalies': Boolean mask of anomalous timesteps
                - 'anomaly_indices': Indices of anomalous timesteps
        """
        if method == "reconstruction":
            errors = self.compute_reconstruction_error(sequence)
            scores = errors.numpy()
        elif method == "latent_velocity":
            # Detect sudden changes in latent space
            trajectory = self.infer_latent_trajectory(sequence)
            velocities = torch.diff(trajectory, dim=0).norm(dim=1)
            # Pad to match original length
            scores = torch.cat([velocities[:1], velocities]).numpy()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute threshold
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold_value = mean_score + threshold * std_score

        anomalies = scores > threshold_value
        anomaly_indices = np.where(anomalies)[0]

        return {
            "scores": scores,
            "threshold": threshold_value,
            "anomalies": anomalies,
            "anomaly_indices": anomaly_indices,
            "mean": mean_score,
            "std": std_score,
        }

    def plot_anomalies(
        self,
        sequence: Tensor,
        anomaly_result: Dict,
        feature_idx: int = 0,
        feature_name: str = "Feature",
        timestamps: Optional[pd.DatetimeIndex] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> Figure:
        """Visualize detected anomalies.

        Args:
            sequence: Observation sequence of shape (T, features).
            anomaly_result: Output from detect_anomalies().
            feature_idx: Index of feature to plot.
            feature_name: Name of the feature.
            timestamps: Optional datetime index for x-axis.
            figsize: Figure size.

        Returns:
            Matplotlib figure with anomaly visualization.
        """
        T = len(sequence)
        x_axis = timestamps if timestamps is not None else np.arange(T)

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot 1: Feature values with anomalies highlighted
        ax1 = axes[0]
        feature_values = sequence[:, feature_idx].numpy()

        ax1.plot(x_axis, feature_values, "b-", alpha=0.7, label=feature_name)

        # Highlight anomalies
        anomaly_mask = anomaly_result["anomalies"]
        if isinstance(x_axis, pd.DatetimeIndex):
            anomaly_x = x_axis[anomaly_mask]
        else:
            anomaly_x = x_axis[anomaly_mask]

        ax1.scatter(
            anomaly_x,
            feature_values[anomaly_mask],
            c="red",
            s=50,
            zorder=5,
            label=f"Anomalies (n={anomaly_mask.sum()})",
        )

        ax1.set_ylabel(feature_name)
        ax1.set_title(f"{feature_name} with Detected Anomalies")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Anomaly scores
        ax2 = axes[1]
        scores = anomaly_result["scores"]

        ax2.plot(x_axis, scores, "g-", alpha=0.7, label="Anomaly Score")
        ax2.axhline(
            anomaly_result["threshold"],
            color="red",
            linestyle="--",
            label=f"Threshold ({anomaly_result['threshold']:.2f})",
        )
        ax2.fill_between(
            x_axis,
            scores,
            anomaly_result["threshold"],
            where=scores > anomaly_result["threshold"],
            color="red",
            alpha=0.3,
        )

        ax2.set_xlabel("Time" if timestamps is None else "Date")
        ax2.set_ylabel("Anomaly Score")
        ax2.set_title("Anomaly Scores Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if timestamps is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def plot_regime_timeline(
        self,
        sequence: Tensor,
        regime_result: Dict,
        seq_idx: int = 0,
        feature_idx: int = 0,
        feature_name: str = "Feature",
        timestamps: Optional[pd.DatetimeIndex] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> Figure:
        """Plot regime assignments over time alongside a feature.

        Args:
            sequence: Observation sequence of shape (T, features).
            regime_result: Output from detect_regimes().
            seq_idx: Index of the sequence in the regime result.
            feature_idx: Index of feature to plot.
            feature_name: Name of the feature.
            timestamps: Optional datetime index for x-axis.
            figsize: Figure size.

        Returns:
            Matplotlib figure with regime timeline.
        """
        # Extract labels for this sequence
        lengths = regime_result["lengths"]
        start_idx = sum(lengths[:seq_idx])
        end_idx = start_idx + lengths[seq_idx]
        labels = regime_result["labels"][start_idx:end_idx]

        T = len(sequence)
        x_axis = timestamps if timestamps is not None else np.arange(T)

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        n_regimes = len(np.unique(regime_result["labels"]))
        cmap = plt.get_cmap("Set2")
        colors = cmap(np.linspace(0, 1, n_regimes))

        # Plot 1: Feature values colored by regime
        ax1 = axes[0]
        feature_values = sequence[:, feature_idx].numpy()

        for regime in range(n_regimes):
            mask = labels == regime
            if isinstance(x_axis, pd.DatetimeIndex):
                regime_x = x_axis[mask]
            else:
                regime_x = x_axis[mask]

            ax1.scatter(
                regime_x,
                feature_values[mask],
                c=[colors[regime]],
                s=15,
                label=f"Regime {regime}",
                alpha=0.7,
            )

        ax1.set_ylabel(feature_name)
        ax1.set_title(f"{feature_name} by Regime")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Regime timeline
        ax2 = axes[1]

        # Create colored background for each regime
        for i in range(len(labels)):
            regime = labels[i]
            if isinstance(x_axis, pd.DatetimeIndex):
                x_start = x_axis[i]
                if i + 1 < len(x_axis):
                    x_end = x_axis[i + 1]
                else:
                    x_end = x_start + (x_axis[1] - x_axis[0])
                ax2.axvspan(x_start, x_end, color=colors[regime], alpha=0.7)
            else:
                ax2.axvspan(i, i + 1, color=colors[regime], alpha=0.7)

        ax2.set_xlabel("Time" if timestamps is None else "Date")
        ax2.set_ylabel("Regime")
        ax2.set_yticks([])
        ax2.set_title("Regime Timeline")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=f"Regime {i}")
            for i in range(n_regimes)
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        if timestamps is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def summarize_dynamics(
        self,
        sequences: List[Tensor],
        feature_names: Tuple[str, ...],
        n_regimes: int = 3,
    ) -> Dict:
        """Generate a comprehensive summary of learned dynamics.

        Args:
            sequences: List of observation sequences.
            feature_names: Names of the features.
            n_regimes: Number of regimes to detect.

        Returns:
            Dictionary with analysis results.
        """
        print("Analyzing hidden dynamics...")

        # 1. Detect regimes
        print("  - Detecting pollution regimes...")
        regime_result = self.detect_regimes(sequences, n_regimes=n_regimes)

        # 2. Compute feature-latent correlations
        print("  - Computing feature-latent correlations...")
        correlations = self.compute_feature_latent_correlation(sequences)
        correlations.index = feature_names

        # 3. Compute regime characteristics
        labels = regime_result["labels"]
        all_obs = torch.cat(sequences, dim=0).numpy()

        regime_characteristics = []
        for regime in range(n_regimes):
            mask: NDArray[np.bool_] = labels == regime
            regime_obs = all_obs[mask]
            char: Dict[str, Any] = {
                "regime": regime,
                "count": int(np.sum(mask)),
                "percentage": float(np.sum(mask) / len(labels) * 100),
            }
            for i, name in enumerate(feature_names):
                char[f"{name}_mean"] = float(regime_obs[:, i].mean())
                char[f"{name}_std"] = float(regime_obs[:, i].std())
            regime_characteristics.append(char)

        # 4. Identify dominant latent dimensions
        top_correlations: Dict[str, Tuple[str, float]] = {}
        for feature in feature_names:
            corr_row = correlations.loc[feature].abs()
            top_dim = str(corr_row.idxmax())
            top_corr_val: Any = correlations.at[feature, top_dim]
            top_correlations[feature] = (top_dim, float(top_corr_val))

        print("  - Analysis complete!")

        return {
            "regime_result": regime_result,
            "correlations": correlations,
            "regime_characteristics": pd.DataFrame(regime_characteristics),
            "top_correlations": top_correlations,
            "n_sequences": len(sequences),
            "total_timesteps": len(labels),
        }
