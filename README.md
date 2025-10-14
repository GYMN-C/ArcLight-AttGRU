# ArcLight-AttGRU: Ultra-Lightweight Attention-Gated GRU for Low-Frequency Arc Fault Detection

This repository contains the code for **ArcLight-AttGRU**, an ultra-lightweight attention-gated GRU framework for low-frequency arc fault detection, as described in the paper **"ArcLight-AttGRU: An Ultra-Lightweight Attention-Gated GRU Framework for Low-Frequency Arc Fault Detection"**, to be published in *IEEE Transactions on Instrumentation and Measurement (TIM)*.

ArcLight-AttGRU leverages **Attention Mechanisms** and **Gated Recurrent Units (GRU)** to efficiently detect arc faults in electrical systems, with minimal computational overhead, making it suitable for embedded real-time applications.

## Key Features

- **Lightweight Model**: Optimized for embedded systems with low memory and power constraints.
- **Attention Mechanism**: Enhances fault detection by focusing on critical time steps.
- **Pruning & Optimization**: Model pruning for reduced size and faster inference.
- **Interpretability**: Includes techniques like **Integrated Gradients** and **Ablation** to explain model decisions.

## Project Structure

- `ArcLight-AttGRU.py`: Main script implementing model training, evaluation, and interpretability modules.
- `lstm/`: Contains the dataset (CSV format) for training, validation, and testing.
- `ablation_plots/`: Stores ablation analysis plots.

## Requirements

- TensorFlow >= 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- tensorflow_model_optimization

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
