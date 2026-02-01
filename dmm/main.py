"""Main entry point for the Deep Markov Model.

This module provides the training and forecasting pipeline for the DMM
using real data from data.csv.
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
    preprocess_dataframe,
    build_sequences,
    sequences_to_tensors,
    train_test_split_sequences,
    plot_training_loss,
    plot_forecast,
)


def load_and_preprocess_data(
    filepath: str, data_config: DataConfig
) -> pd.DataFrame:
    """Load and preprocess the dataset from CSV.

    Args:
        filepath: Path to the CSV file.
        data_config: Data configuration.

    Returns:
        Preprocessed DataFrame with normalized values.
    """
    # Load CSV file
    df = pd.read_csv(filepath)

    # Convert datetime - handle mixed formats
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, format="mixed")

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # Convert numeric columns (handle European decimal separator)
    for col in data_config.columns:
        if col in df.columns:
            # Convert to string and replace comma with dot for decimal
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            # Handle missing value indicators
            df[col] = df[col].replace(["N/D", "-", "nan", "None"], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract only the columns we need
    df_subset = df[["datetime"] + list(data_config.columns)].copy()

    # Preprocess (normalize and fill short NaN gaps)
    df_processed, norm_stats = preprocess_dataframe(
        df_subset[list(data_config.columns)],
        data_config.columns,
        data_config.max_nan_gap,
    )

    # Add datetime back for reference
    df_processed["datetime"] = df_subset["datetime"].values

    return df_processed, norm_stats


def main():
    """Main function for training DMM and forecasting."""
    # Clear Pyro param store
    pyro.clear_param_store()

    # Check for quick test mode (for debugging)
    quick_test = os.environ.get("DMM_QUICK_TEST", "0") == "1"

    # Configuration
    model_config = ModelConfig(
        z_dim=16,
        transition_dim=64,
        emission_dim=32,
        rnn_dim=64,
        output_dim=5,  # PM10, temp, humidity, rain, wind
    )

    training_config = TrainingConfig(
        lr=1e-3,
        num_epochs=50 if quick_test else 500,
        print_every=10 if quick_test else 50,
        random_seed=42,
    )

    data_config = DataConfig(
        max_nan_gap=2,
        test_size=0.2,
        columns=("PM10", "temp", "humidity", "rain", "wind"),
    )

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, "data.csv")

    # Create organized output directory structure
    output_dir = os.path.join(project_dir, "analysis_results")
    figures_training_dir = os.path.join(output_dir, "figures", "training")
    figures_forecasts_dir = os.path.join(output_dir, "figures", "forecasts")
    data_forecasts_dir = os.path.join(output_dir, "data", "forecasts")

    os.makedirs(figures_training_dir, exist_ok=True)
    os.makedirs(figures_forecasts_dir, exist_ok=True)
    os.makedirs(data_forecasts_dir, exist_ok=True)

    print("=" * 60)
    print("Deep Markov Model - Training Pipeline")
    print("=" * 60)

    # Load and preprocess data
    print(f"\nLoading data from: {data_path}")
    df, norm_stats = load_and_preprocess_data(data_path, data_config)
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Features: {list(data_config.columns)}")

    # Build sequences from contiguous data blocks
    print("\nBuilding sequences from contiguous data blocks...")
    feature_cols = list(data_config.columns)
    sequences_list = build_sequences(
        df[feature_cols],
        min_length=24,  # Minimum 24 hours (1 day) per sequence
    )
    print(f"Built {len(sequences_list)} sequences")

    if len(sequences_list) == 0:
        print("ERROR: No valid sequences found. Check data for NaN gaps.")
        return

    # Print sequence statistics
    seq_lengths = [len(s) for s in sequences_list]
    print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
          f"mean={np.mean(seq_lengths):.1f}")

    # Convert to tensors
    sequences = sequences_to_tensors(sequences_list)

    # Split into train/test
    train_seqs, test_seqs = train_test_split_sequences(
        sequences,
        test_size=data_config.test_size,
        random_seed=training_config.random_seed,
    )

    print(f"\nTraining sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")

    # Create model using factory
    print("\nCreating Deep Markov Model...")
    model = DMMFactory.create_default(model_config)
    print(f"  Latent dimension: {model_config.z_dim}")
    print(f"  RNN dimension: {model_config.rnn_dim}")
    print(f"  Output dimension: {model_config.output_dim}")

    # Create trainer
    trainer = DMMTrainer(model, training_config)

    # Train the model
    print(f"\nTraining for {training_config.num_epochs} epochs...")
    print("-" * 40)
    history = trainer.fit(
        train_seqs,
        test_seqs,
        num_epochs=training_config.num_epochs,
        print_every=training_config.print_every,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final train ELBO: {history['train_losses'][-1]:.4f}")
    final_test_loss = trainer.evaluate(test_seqs)
    print(f"Final test ELBO: {final_test_loss:.4f}")

    # Forecasting
    print("\n" + "=" * 60)
    print("Forecasting Future Values")
    print("=" * 60)

    forecaster = DMMForecaster(model)

    # Use the longest test sequence for inference
    test_seq = max(test_seqs, key=lambda x: x.shape[0])
    print(f"\nUsing test sequence of length {test_seq.shape[0]} hours")

    # Infer latent state from observations
    z_last = forecaster.infer_latent(test_seq)
    print(f"Inferred latent state shape: {z_last.shape}")

    # Forecast next 7 days (168 hours)
    forecast_hours = 168  # 7 days
    print(f"\nForecasting next {forecast_hours} hours ({forecast_hours // 24} days)...")

    forecast_samples = forecaster.forecast(
        z_last,
        steps=forecast_hours,
        num_samples=100,
    )
    mean, lower, upper = forecaster.get_prediction_intervals(
        forecast_samples,
        lower_quantile=0.1,
        upper_quantile=0.9,
    )

    print(f"Forecast samples shape: {forecast_samples.shape}")
    print(f"Mean forecast shape: {mean.shape}")

    # Denormalize forecasts to original scale
    mean_denorm = norm_stats.denormalize_all(mean.numpy(), data_config.columns)
    lower_denorm = norm_stats.denormalize_all(lower.numpy(), data_config.columns)
    upper_denorm = norm_stats.denormalize_all(upper.numpy(), data_config.columns)

    # Print forecast summary for each feature (in original scale)
    print("\nForecast Summary (mean values for first 24 hours, original scale):")
    print("-" * 60)
    for i, col in enumerate(data_config.columns):
        mean_val = mean_denorm[:24, i].mean()
        lower_val = lower_denorm[:24, i].mean()
        upper_val = upper_denorm[:24, i].mean()
        orig_mean, orig_std = norm_stats.get(col)
        print(f"  {col}: {mean_val:.2f} (90% CI: [{lower_val:.2f}, {upper_val:.2f}]) "
              f"[original: mean={orig_mean:.2f}, std={orig_std:.2f}]")

    # Generate plots
    print("\nGenerating plots...")

    # Plot training loss
    fig_loss = plot_training_loss(
        history["train_losses"],
        test_loss=final_test_loss,
        smooth_window=10,
    )
    loss_path = os.path.join(figures_training_dir, "training_loss.png")
    fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {loss_path}")

    # Denormalize test sequence for plotting
    test_seq_denorm = torch.tensor(
        norm_stats.denormalize_all(test_seq.numpy(), data_config.columns),
        dtype=torch.float32,
    )
    mean_denorm_t = torch.tensor(mean_denorm, dtype=torch.float32)
    lower_denorm_t = torch.tensor(lower_denorm, dtype=torch.float32)
    upper_denorm_t = torch.tensor(upper_denorm, dtype=torch.float32)

    # Plot forecast for each feature (in original scale)
    for i, col in enumerate(data_config.columns):
        fig_forecast = plot_forecast(
            test_seq_denorm,
            mean_denorm_t,
            lower_denorm_t,
            upper_denorm_t,
            feature_idx=i,
            feature_name=col,
        )
        forecast_path = os.path.join(figures_forecasts_dir, f"forecast_{col}.png")
        fig_forecast.savefig(forecast_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {forecast_path}")

    # Save forecast to CSV (in original scale)
    forecast_df = pd.DataFrame(
        mean_denorm,
        columns=[f"{col}_forecast" for col in data_config.columns],
    )
    forecast_df["hour"] = range(1, forecast_hours + 1)
    for i, col in enumerate(data_config.columns):
        forecast_df[f"{col}_lower"] = lower_denorm[:, i]
        forecast_df[f"{col}_upper"] = upper_denorm[:, i]

    forecast_csv_path = os.path.join(data_forecasts_dir, "forecast_results.csv")
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"Saved: {forecast_csv_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
