# ArcLight-AttGRU: Ultra-Lightweight Attention-Gated GRU for Low-Frequency Arc Fault Detection


ArcLight-AttGRU combines **Attention Mechanisms** and **Gated Recurrent Units (GRU)** to efficiently detect arc faults in electrical systems. The model is designed to be ultra-lightweight, offering minimal computational overhead while achieving high detection accuracy, making it suitable for real-time embedded applications with limited resources such as memory and processing power.

## Key Contributions

- **Attention-Gated GRU**: Combines the attention mechanism and GRU to effectively detect low-frequency arc faults while maintaining efficiency.
- **Model Pruning**: Optimizes the model by pruning unnecessary parameters, reducing memory usage and improving inference time.
- **Interpretability**: Implements **Integrated Gradients** and **Ablation** techniques to provide interpretability for model decisions.
- **Real-Time Fault Detection**: Designed for real-time embedded systems, ensuring low latency and high reliability.

## Project Structure

- `ArcLight-AttGRU.py`: The main script implementing model training, evaluation, pruning, and interpretability functions.
- `lstm/`: Directory containing training, validation, and test datasets in CSV format.
- `ablation_plots/`: Directory where the ablation analysis plots are saved.

## Requirements

To run this project, you need the following dependencies:

- **TensorFlow** >= 2.x: For building and training the neural network.
- **NumPy**: For numerical computing and data handling.
- **Pandas**: For data manipulation and CSV file processing.
- **Scikit-learn**: For evaluation metrics and clustering.
- **Matplotlib**: For plotting graphs, particularly for ablation analysis.
- **tensorflow_model_optimization**: For model pruning and optimization.
- **Joblib**: For parallel processing and handling large datasets.

### Install Dependencies

Install all required dependencies by running:

```bash
pip install -r requirements.txt
````

## How to Run

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/ArcLight-AttGRU.git
```

### Prepare the Data

Make sure your dataset is in CSV format and placed in the `lstm/` directory. The dataset should include features for training, validation, and testing. You can use the provided datasets or replace them with your own dataset formatted similarly.

### Train the Model

To train the model, run the following command:

```bash
python ArcLight-AttGRU.py
```

This will:

* Load the training and validation data.
* Train the ArcLight-AttGRU model using Attention-Gated GRU.
* Apply pruning to the model for optimization.
* Log key metrics such as accuracy, precision, recall, and F1 score during training.
* Save the pruned model and evaluation results.

### Model Interpretability

For a better understanding of model decisions, this project includes two interpretability methods:

* **Integrated Gradients**: This method helps visualize the importance of each feature in the model's prediction.
* **Ablation Analysis**: This technique removes each time step in the input data to analyze its impact on the model's output. The results are saved as PNG images in the `ablation_plots/` directory.

---

## Project 2: Motifs

In addition to the **ArcLight-AttGRU**, this repository also includes **Motifs**, a tool for extracting multi-scale features and performing clustering on time-series data.

### Key Features of Motifs

* **Multi-Scale Feature Extraction**: Extracts statistical features like mean, min, max, and standard deviation at multiple time scales.
* **KMeans Clustering**: Groups time-series data based on extracted features to discover patterns.
* **Data Saving**: Saves processed features and labels for further analysis in `.pkl` files.

### How to Run Motifs

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/Time-Series-Analysis-Projects.git
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**: Place your time-series data files in the `temporal_patterns/` directory.

4. **Run the main script**:

   ```bash
   python motifs.py
   ```

This will:

* Load the time-series data from the specified directory.
* Extract multi-scale features from the data.
* Perform KMeans clustering to identify patterns.
* Save the processed data for future use.

---

## Dataset Information

The datasets used for training and evaluating the ArcLight-AttGRU model were obtained from **IEEE DataPort** (DOI: [10.21227/fwed-0w934](https://ieeexplore.ieee.org/document/9274571)).

Please cite the dataset if you use it in your research:

```
IEEE DataPort. (2020). IEEE DataPort (DOI: 10.21227/fwed-0w934). Retrieved from [https://ieeexplore.ieee.org/document/9274571](https://ieeexplore.ieee.org/document/9274571)
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

We would like to thank the original authors of the **GRU** and **Attention Mechanism** models, whose work laid the foundation for this project. Additionally, we appreciate the support of the open-source community for providing the tools necessary to bring this project to life.

---

## Further Information

For any questions or issues, feel free to contact us at [[gyn13944041446@outlook.com](mailto:gyn13944041446@outlook.com)]. We welcome contributions from the community to improve this framework.

