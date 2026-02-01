"""Analysis script for understanding DMM hidden dynamics.

This script trains a DMM model and then uses the DMMInterpreter
to analyze the hidden dynamics in air quality data.
"""

import os
import numpy as np
import pandas as pd
import torch
import pyro

from dmm import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    DMMFactory,
    DMMTrainer,
    DMMForecaster,
    DMMInterpreter,
    preprocess_dataframe,
    build_sequences,
    sequences_to_tensors,
    train_test_split_sequences,
)


def load_and_preprocess_data(filepath: str, data_config: DataConfig):
    """Load and preprocess the dataset from CSV."""
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, format="mixed")
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in data_config.columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = df[col].replace(["N/D", "-", "nan", "None"], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_subset = df[["datetime"] + list(data_config.columns)].copy()

    df_processed, norm_stats = preprocess_dataframe(
        df_subset[list(data_config.columns)],
        data_config.columns,
        data_config.max_nan_gap,
    )

    df_processed["datetime"] = df_subset["datetime"].values
    return df_processed, norm_stats


def main():
    """Main analysis function."""
    pyro.clear_param_store()

    # Check for quick test mode
    quick_test = os.environ.get("DMM_QUICK_TEST", "0") == "1"

    # Configuration
    model_config = ModelConfig(
        z_dim=16,
        transition_dim=64,
        emission_dim=32,
        rnn_dim=64,
        output_dim=5,
    )

    training_config = TrainingConfig(
        lr=1e-3,
        num_epochs=50 if quick_test else 200,
        print_every=10 if quick_test else 25,
        random_seed=42,
    )

    data_config = DataConfig(
        max_nan_gap=2,
        test_size=0.2,
        columns=("PM10", "temp", "humidity", "rain", "wind"),
    )

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, "data.csv")

    # Create organized output directory structure
    output_dir = os.path.join(project_dir, "analysis_results")
    figures_analysis_dir = os.path.join(output_dir, "figures", "analysis")
    data_analysis_dir = os.path.join(output_dir, "data", "analysis")

    os.makedirs(figures_analysis_dir, exist_ok=True)
    os.makedirs(data_analysis_dir, exist_ok=True)

    print("=" * 70)
    print("Deep Markov Model - Hidden Dynamics Analysis")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df, norm_stats = load_and_preprocess_data(data_path, data_config)
    print(f"Loaded {len(df)} records")

    # Build sequences
    print("\nBuilding sequences...")
    feature_cols = list(data_config.columns)
    sequences_list = build_sequences(df[feature_cols], min_length=24)
    print(f"Built {len(sequences_list)} sequences")

    sequences = sequences_to_tensors(sequences_list)
    train_seqs, test_seqs = train_test_split_sequences(
        sequences, test_size=data_config.test_size, random_seed=42
    )

    # Train model
    print(f"\n{'=' * 70}")
    print("PHASE 1: Training Deep Markov Model")
    print("=" * 70)

    model = DMMFactory.create_default(model_config)
    trainer = DMMTrainer(model, training_config)

    print(f"\nTraining for {training_config.num_epochs} epochs...")
    history = trainer.fit(
        train_seqs, test_seqs,
        num_epochs=training_config.num_epochs,
        print_every=training_config.print_every,
    )

    print(f"\nFinal train ELBO: {history['train_losses'][-1]:.4f}")
    print(f"Final test ELBO: {trainer.evaluate(test_seqs):.4f}")

    # Analysis
    print(f"\n{'=' * 70}")
    print("PHASE 2: Analyzing Hidden Dynamics")
    print("=" * 70)

    forecaster = DMMForecaster(model)
    interpreter = DMMInterpreter(model, forecaster)

    # Use all sequences for analysis
    all_seqs = train_seqs + test_seqs

    # 1. Summarize dynamics
    print("\n--- Comprehensive Dynamics Summary ---")
    summary = interpreter.summarize_dynamics(
        all_seqs,
        feature_names=data_config.columns,
        n_regimes=3,
    )

    print(f"\nAnalyzed {summary['n_sequences']} sequences "
          f"({summary['total_timesteps']} total timesteps)")

    print("\n>>> Regime Characteristics:")
    print(summary["regime_characteristics"].to_string(index=False))

    print("\n>>> Top Feature-Latent Correlations:")
    for feature, (dim, corr) in summary["top_correlations"].items():
        print(f"  {feature}: {dim} (r = {corr:+.3f})")

    # 2. Correlation heatmap
    print("\n--- Feature-Latent Correlation Analysis ---")
    fig_corr = interpreter.plot_feature_latent_correlation(
        all_seqs, data_config.columns
    )
    corr_path = os.path.join(figures_analysis_dir, "feature_latent_correlation.png")
    fig_corr.savefig(corr_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {corr_path}")

    # 3. Regime characteristics
    print("\n--- Regime Analysis ---")
    regime_result = summary["regime_result"]

    fig_regimes = interpreter.plot_regime_characteristics(
        all_seqs, regime_result, data_config.columns
    )
    regimes_path = os.path.join(figures_analysis_dir, "regime_characteristics.png")
    fig_regimes.savefig(regimes_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {regimes_path}")

    # 4. Latent trajectory for longest sequence
    print("\n--- Latent Space Trajectory ---")
    longest_seq = max(all_seqs, key=lambda x: x.shape[0])
    pm10_values = longest_seq[:, 0]  # PM10 is first column

    # Denormalize PM10 for coloring
    pm10_denorm = torch.tensor(
        norm_stats.denormalize(pm10_values.numpy(), "PM10"),
        dtype=torch.float32
    )

    fig_traj = interpreter.plot_latent_trajectory(
        longest_seq,
        method="pca",
        feature_name="PM10 (µg/m³)",
        feature_values=pm10_denorm,
    )
    traj_path = os.path.join(figures_analysis_dir, "latent_trajectory.png")
    fig_traj.savefig(traj_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {traj_path}")

    # 5. Regime timeline
    print("\n--- Regime Timeline ---")
    # Find the index of longest_seq in all_seqs
    seq_idx = 0
    for i, seq in enumerate(all_seqs):
        if seq.shape[0] == longest_seq.shape[0]:
            seq_idx = i
            break

    fig_timeline = interpreter.plot_regime_timeline(
        longest_seq,
        regime_result,
        seq_idx=seq_idx,
        feature_idx=0,
        feature_name="PM10 (normalized)",
    )
    timeline_path = os.path.join(figures_analysis_dir, "regime_timeline.png")
    fig_timeline.savefig(timeline_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {timeline_path}")

    # 6. Anomaly detection
    print("\n--- Anomaly Detection ---")
    anomaly_result = interpreter.detect_anomalies(
        longest_seq, threshold=2.5, method="reconstruction"
    )
    print(f"Detected {anomaly_result['anomalies'].sum()} anomalies "
          f"(threshold = {anomaly_result['threshold']:.2f})")

    fig_anomalies = interpreter.plot_anomalies(
        longest_seq,
        anomaly_result,
        feature_idx=0,
        feature_name="PM10 (normalized)",
    )
    anomalies_path = os.path.join(figures_analysis_dir, "anomaly_detection.png")
    fig_anomalies.savefig(anomalies_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {anomalies_path}")

    # 7. Save detailed results
    print("\n--- Saving Detailed Results ---")

    # Save correlations
    corr_csv_path = os.path.join(data_analysis_dir, "feature_latent_correlations.csv")
    summary["correlations"].to_csv(corr_csv_path)
    print(f"Saved: {corr_csv_path}")

    # Save regime characteristics
    regime_csv_path = os.path.join(data_analysis_dir, "regime_characteristics.csv")
    summary["regime_characteristics"].to_csv(regime_csv_path, index=False)
    print(f"Saved: {regime_csv_path}")

    # Interpretation summary
    print(f"\n{'=' * 70}")
    print("INTERPRETATION SUMMARY")
    print("=" * 70)

    print("\n>>> Key Findings:")

    # Interpret regimes
    regime_chars = summary["regime_characteristics"]
    pm10_means = [regime_chars.iloc[i]["PM10_mean"] for i in range(3)]
    sorted_regimes = np.argsort(pm10_means)

    print("\n  Pollution Regimes (sorted by PM10 level):")
    for rank, regime_idx in enumerate(sorted_regimes):
        regime = regime_chars.iloc[regime_idx]
        level = "LOW" if rank == 0 else ("HIGH" if rank == 2 else "MODERATE")
        print(f"    Regime {regime['regime']} ({level} pollution):")
        print(f"      - PM10: {regime['PM10_mean']:.2f} (normalized)")
        print(f"      - Temperature: {regime['temp_mean']:.2f}")
        print(f"      - Humidity: {regime['humidity_mean']:.2f}")
        print(f"      - Wind: {regime['wind_mean']:.2f}")
        print(f"      - Frequency: {regime['percentage']:.1f}% of time")

    # Interpret correlations
    print("\n  Latent Dimension Interpretations:")
    for feature, (dim, corr) in summary["top_correlations"].items():
        direction = "positively" if corr > 0 else "negatively"
        print(f"    {dim} is {direction} correlated with {feature} (r={corr:+.3f})")

    print(f"\n{'=' * 70}")
    print("Analysis complete! Results saved to:")
    print(f"  {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
