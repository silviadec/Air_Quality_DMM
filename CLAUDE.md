# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Markov Model (DMM) implementation for probabilistic time series modeling, built with PyTorch and Pyro. Based on the Pyro DMM example (https://pyro.ai/examples/dmm.html).

## Commands

```bash
# Install dependencies (use conda environment)
pip install -r requirements.txt

# Run with specific Python environment
/Users/silviobaratto/anaconda3/envs/dmm/bin/python dmm/main.py

# Quick import test
python -c "from dmm import ModelConfig, DMMFactory, DMMTrainer"
```

## Architecture

The codebase follows SOLID principles with dependency injection:

### Core Components

**Model Layer** (`dmm/models/`):
- `base.py`: Abstract interfaces (`BaseTransition`, `BaseEmitter`, `BaseCombiner`) - all return `Tuple[Tensor, Tensor]` for (loc, scale)
- `dmm.py`: `DeepMarkovModel(PyroModule)` accepts injected components via constructor
- `factory.py`: `DMMFactory` creates pre-configured models from `ModelConfig`

**Configuration** (`dmm/config.py`):
- `ModelConfig`: Architecture dims (z_dim, transition_dim, emission_dim, rnn_dim, output_dim)
- `TrainingConfig`: Training hyperparameters (lr, epochs, clip_norm)
- `DataConfig`: Preprocessing settings

**Training** (`dmm/training/trainer.py`):
- `DMMTrainer`: Wraps Pyro SVI with Trace_ELBO loss
- Methods: `train_epoch()`, `evaluate()`, `fit()`

**Inference** (`dmm/inference/forecaster.py`):
- `DMMForecaster`: Latent inference and future prediction
- Methods: `infer_latent()`, `forecast()`, `get_prediction_intervals()`

**Data Pipeline** (`dmm/data/`):
- `preprocessing.py`: z-score normalization, NaN interpolation
- `dataset.py`: Build contiguous sequences from DataFrame, `SequenceDataset` wrapper

### DMM Mathematical Structure

The model defines:
- **Transition prior**: `p(z_t | z_{t-1})` via `GatedTransition`
- **Emission likelihood**: `p(x_t | z_t)` via `Emitter`
- **Approximate posterior**: `q(z_t | z_{t-1}, h_rnn)` via `Combiner`

All distributions are diagonal Gaussians parameterized by neural networks.

## Usage Pattern

```python
from dmm import ModelConfig, DMMFactory, DMMTrainer, DMMForecaster

# Create model
config = ModelConfig(z_dim=16, output_dim=5)
model = DMMFactory.create_default(config)

# Train
trainer = DMMTrainer(model)
history = trainer.fit(train_seqs, test_seqs)

# Forecast
forecaster = DMMForecaster(model)
z_last = forecaster.infer_latent(sequence)
predictions = forecaster.forecast(z_last, steps=24)
```

## Type Hints

The codebase uses strict typing. For Pyro/pandas type issues, use `cast(Any, value)` pattern:
```python
from typing import Any, cast
loss = float(cast(Any, self.svi.step(seq, length)))
```
