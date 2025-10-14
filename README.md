

````markdown
# ArcLight-AttGRU: An Ultra-Lightweight Attention-Gated GRU Framework for Low-Frequency Arc Fault Detection


The ArcLight-AttGRU model combines **Attention Mechanisms** and **Gated Recurrent Units (GRU)** to detect low-frequency arc faults in electrical systems. This model is designed to be **ultra-lightweight**, ensuring high detection accuracy while minimizing computational cost, making it suitable for real-time, embedded systems where resource constraints are critical.

## Overview

ArcLight-AttGRU focuses on improving the detection accuracy and computational efficiency of arc fault detection systems, particularly for low-frequency arc faults. By using attention mechanisms in combination with GRUs, we achieve higher performance while maintaining low power and memory usage, which is crucial for industrial and embedded applications.

### Contributions

- **Attention-Gated GRU Framework**: A lightweight, efficient framework designed to enhance arc fault detection capabilities with minimal computational overhead.
- **Integrated Gradients & Ablation**: The model provides interpretability through methods like Integrated Gradients and Ablation to help understand model decisions.
- **Low-Resource Deployment**: Optimized for embedded systems and real-time fault detection with reduced model size and fast inference times.

## Project Structure

- `ArcLight-AttGRU.py`: The main script implementing the ArcLight-AttGRU framework, including model definition, training, pruning, evaluation, and interpretability modules.
- `lstm/`: Directory containing the training, validation, and testing data in CSV format.
- `ablation_plots/`: Directory where ablation importance plots are saved to visualize the contribution of each time step to the model's predictions.

## Requirements

- TensorFlow >= 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- tensorflow_model_optimization

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
````

## How to Run

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/ArcLight-AttGRU.git
```

### Prepare the Data

Make sure to place your dataset in the `lstm/` directory (the dataset should be in CSV format).

### Train the Model

Run the `ArcLight-AttGRU.py` script to train the model:

```bash
python ArcLight-AttGRU.py
```

This script will:

1. Load the training and validation data.
2. Train the model using the Attention-Gated GRU framework.
3. Apply model pruning for further optimization.
4. Log training results, including metrics like accuracy, precision, recall, and F1 score.

### Interpretability

To better understand model predictions, the following interpretability techniques are included:

* **Integrated Gradients**: Provides insight into the importance of each feature in the model's decision-making process.
* **Ablation Analysis**: Visualizes the effect of removing different time steps from the input sequence.

Results will be saved in the `ablation_plots/` directory as PNG images for further analysis.

---

## Project 2: Motifs

In addition to ArcLight-AttGRU, this repository includes **Motifs**, a tool for extracting multi-scale features and performing clustering analysis on time-series data. It is useful for exploring underlying patterns in time-series sequences.

### Motifs Overview

**Motifs** extracts statistical features from time-series data at multiple scales and uses KMeans clustering to identify patterns. This tool is intended for users interested in discovering meaningful patterns within time-series data, such as those from sensor networks, industrial systems, or biological data.

### Motifs Workflow

1. **Data Loading**: The system automatically loads time-series data from CSV files in the `temporal_patterns/` directory.
2. **Multi-Scale Feature Extraction**: Features such as mean, min, max, and standard deviation are extracted at various time scales.
3. **Clustering**: KMeans clustering is applied to the extracted features to group similar patterns in the data.
4. **Saving Processed Data**: The processed features and labels are saved as `.pkl` files for further analysis.


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


