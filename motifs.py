import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pickle

# Load time-series pattern segments (keep original directory traversal logic)
def load_temporal_patterns(base_path):
    """
    Load time-series pattern data by traversing the directory structure.
    :param base_path: The root path containing subdirectories for different classes
    :return: Temporal patterns dataset
    """
    temporal_patterns = []
    
    # Traverse through the binary range from 0b01 to 0b1010
    for i in range(0b01, 0b1011):
        dir_name = bin(i)[2:]  # Convert to binary string (without "0b" prefix)
        dir_path = os.path.join(base_path, dir_name)
        
        # Check if the directory exists
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist, skipping")
            continue
            
        # Read each pattern file in the subdirectory (keep original logic of 20 files)
        for j in range(0, 20):
            file_path = os.path.join(dir_path, f'temporal_patterns_{j}.csv')
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist, skipping")
                continue
                
            try:
                # Read the CSV file
                data = pd.read_csv(file_path, index_col=0)
                temporal_patterns.append(data.values)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return np.array(temporal_patterns)

# Extract multi-scale pattern features from time series
def extract_multi_scale_features(sequence, scales=(5, 10, 20)):
    """
    Extract multi-scale features from a given time series sequence.
    :param sequence: Input time series (1D array)
    :param scales: List of scales to extract features from
    :return: Multi-scale feature vector
    """
    features = []
    for scale in scales:
        # Check if sequence length is sufficient for the current scale
        if len(sequence) < scale:
            # If sequence is too short, pad with NaN
            padding = np.full(scale - len(sequence), np.nan)
            padded_seq = np.concatenate([sequence, padding])
            windows = padded_seq.reshape(1, -1)
        else:
            # Extract sliding windows for the current scale
            windows = np.lib.stride_tricks.sliding_window_view(sequence, scale)
        
        # Calculate statistical features for each window
        features.extend([
            np.nanmean(windows, axis=1),  # Mean feature (ignores NaN)
            np.nanmin(windows, axis=1),   # Min feature
            np.nanmax(windows, axis=1),   # Max feature
            np.nanstd(windows, axis=1)    # Standard deviation feature
        ])
    return np.concatenate(features)

# Function to save datasets to specified path
def save_datasets(base_path, datasets):
    """
    Save all datasets to the specified path.
    :param base_path: Path to save the datasets
    :param datasets: Dictionary containing datasets
    """
    os.makedirs(base_path, exist_ok=True)
    for name, data in datasets.items():
        file_path = os.path.join(base_path, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved: {file_path}")

# Main processing flow - preserves original variable names
def process_and_save_data(input_path='../temporal_patterns', output_path='./processed_datasets'):
    """
    Main time-series pattern processing pipeline. Preserves original variable names and saves results.
    :param input_path: Path to input data
    :param output_path: Path to save processed datasets
    """
    # 1. Load data (preserve original directory traversal logic)
    pattern_data = load_temporal_patterns(input_path)
    
    if len(pattern_data) == 0:
        raise ValueError("No data loaded, please check input path and directory structure")
    
    # 2. Multi-scale feature extraction (parallel processing)
    print("Extracting multi-scale features...")
    feature_extractor = Parallel(n_jobs=-1, verbose=1)(
        delayed(extract_multi_scale_features)(seq.flatten()) 
        for seq in pattern_data
    )
    features = np.array(feature_extractor)
    
    # 3. Data normalization
    print("Normalizing data...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # 4. Cluster-based feature enhancement
    print("Enhancing features with clustering...")
    cluster_model = KMeans(n_clusters=4, random_state=42, n_init='auto')
    cluster_labels = cluster_model.fit_predict(features_normalized)
    features_with_clusters = np.column_stack([features_normalized, cluster_labels])
    
    # 5. Create DataFrame
    features_df = pd.DataFrame(features_with_clusters)
    
    # 6. Set target variable (last column is the target)
    y = features_df.iloc[:, -1]
    X = features_df.iloc[:, :-1]
    
    # 7. Split dataset
    print("Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 8. Preserve original variable names
    global x_train_v2, x_valid, x_test_v2, y_train_v2, y_valid, y_test_v2
    x_train_v2 = X_train
    x_valid = X_valid
    x_test_v2 = X_test
    y_train_v2 = y_train
    y_valid = y_valid
    y_test_v2 = y_test
    
    # Print dataset sizes
    print(f'x_train_v2: {x_train_v2.shape}, y_train_v2: {y_train_v2.shape}')
    print(f'x_valid: {x_valid.shape}, y_valid: {y_valid.shape}')
    print(f'x_test_v2: {x_test_v2.shape}, y_test_v2: {y_test_v2.shape}')
    
    # 9. Save datasets
    datasets = {
        'x_train_v2': x_train_v2,
        'x_valid': x_valid,
        'x_test_v2': x_test_v2,
        'y_train_v2': y_train_v2,
        'y_valid': y_valid,
        'y_test_v2': y_test_v2
    }
    
    save_datasets(output_path, datasets)
    print(f"All datasets saved to: {output_path}")
    
    return datasets

# Execute processing pipeline
if __name__ == "__main__":
    # Use original path names but replace core concepts
    data_dict = process_and_save_data(
        input_path='../temporal_patterns',  # Original path name 'shapelets_DTW_choose'
        output_path='./processed_datasets'
    )
