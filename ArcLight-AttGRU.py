import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, Input, Model, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow_model_optimization.sparsity import keras as sparsity
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score

# ----------------------Interpretability Module: Integrated Gradients----------------------
def integrated_gradients(inputs, model, baseline=None, m_steps=50):
    """
    Calculate the Integrated Gradients for a single sample.
    inputs: np.array, shape=(T, F)
    model: tf.keras.Model, outputs a single sigmoid probability
    baseline: np.array, same shape as inputs, default is zero
    m_steps: number of interpolation steps
    Returns: integrated_grads, same shape as inputs
    """
    if baseline is None:
        baseline = np.zeros_like(inputs)
    # Generate interpolated samples
    interpolated = [
        baseline + (float(k) / m_steps) * (inputs - baseline)
        for k in range(m_steps + 1)
    ]
    grads = []
    for x in interpolated:
        x_tensor = tf.convert_to_tensor(x[np.newaxis, ...], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred = model(x_tensor)[0][0]
        grad = tape.gradient(pred, x_tensor).numpy()[0]
        grads.append(grad)
    avg_grads = np.mean(np.stack(grads, axis=0), axis=0)
    return (inputs - baseline) * avg_grads

# ----------------------Interpretability Module: Ablation----------------------
def ablation_importance(sample, model, baseline=None):
    """
    Ablation for each time step of a single sample:
    - sample: np.array, shape=(T, F)
    - model: tf.keras.Model
    - baseline: np.array, used for masking, default is zero
    Returns: importance, np.array shape=(T,)
    """
    T, F = sample.shape
    if baseline is None:
        baseline = np.zeros_like(sample)
    orig = model.predict(sample[None, ...], verbose=0)[0,0]
    imps = []
    for t in range(T):
        masked = sample.copy()
        masked[t] = baseline[t]
        pred = model.predict(masked[None, ...], verbose=0)[0,0]
        imps.append(orig - pred)
    return np.array(imps)

def visualize_ablation(model, x_data, sample_indices=[0,1,2], save_dir="./ablation_plots"):
    """
    Perform ablation analysis on specific samples and visualize the results.
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx in sample_indices:
        sample = x_data[idx]
        imps = ablation_importance(sample, model)
        plt.figure(figsize=(8,3))
        plt.plot(np.arange(len(imps)), imps, marker='o')
        plt.title(f"Ablation Importance - Sample {idx}")
        plt.xlabel("Time Step")
        plt.ylabel("OrigPred - MaskedPred")
        plt.grid(True)
        path = os.path.join(save_dir, f"ablation_{idx}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Saved ablation plot: {path}")

# ----------------------Model Architecture---------------------------------------
def attention_block(inputs):
    """
    Attention mechanism that weights the inputs based on attention probabilities.
    """
    feature_dim = inputs.shape[-1]
    att_probs = layers.Dense(feature_dim, activation='softmax', name='attention_vec')(inputs)
    return layers.Multiply(name='attention_mul')([inputs, att_probs])

def build_lightweight_model(n_steps_in, n_features):
    """
    Build a lightweight GRU-based model with attention mechanism.
    """
    inp = Input(shape=(n_steps_in, n_features), name="input")
    x = layers.SeparableConv1D(64, 3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.GRU(64, return_sequences=True)(x)
    x = attention_block(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(1, activation='sigmoid', name="output")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')]
    )
    return model

# ----------------------Callbacks---------------------------------------
class EpochLogger(Callback):
    """
    Custom callback to log metrics after each epoch.
    """
    def __init__(self, validation_data, log_file):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.log_file = log_file
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Log loss, accuracy, and other metrics after each epoch.
        """
        loss, acc = logs['loss'], logs['accuracy']
        vloss, vacc = logs['val_loss'], logs['val_accuracy']
        y_prob = self.model.predict(self.x_val, verbose=0)
        y_pred = (y_prob >= 0.5).astype(int).flatten()
        y_true = self.y_val.flatten()
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        line = f"{epoch+1}\t{loss:.4f}\t{acc:.4f}\t{vloss:.4f}\t{vacc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\n"
        self.log_file.write(line)
        self.log_file.flush()
        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, val_f1={f1:.4f}")

class SparsityCallback(Callback):
    """
    Callback to monitor model sparsity during training.
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate and print the sparsity (proportion of zero weights) at the end of each epoch.
        """
        zeros = total = 0
        for layer in self.model.layers:
            for w in layer.get_weights():
                zeros += np.sum(w == 0)
                total += w.size
        print(f"Epoch {epoch+1} sparsity: {zeros / total:.4f}")

# --------------------------Main Process---------------------------------------
def main():
    """
    Main function to load data, train the model, and evaluate performance.
    """
    # Write header for the log file
    log_fp = open("epoch_results.txt", "w")
    log_fp.write("epoch\tloss\tacc\tval_loss\tval_acc\tprec\trec\tf1\n")

    # Load data
    x_train = pd.read_csv("lstm/x_train_v2.csv").iloc[:, 1:51].values.reshape(-1, 50, 1)
    x_val = pd.read_csv("lstm/x_vaild.csv").iloc[:, 1:51].values.reshape(-1, 50, 1)
    x_test = pd.read_csv("lstm/x_test_v2.csv").iloc[:, 1:51].values.reshape(-1, 50, 1)
    y_train = pd.read_csv("lstm/y_train_v2.csv").iloc[:, 1:].values
    y_val = pd.read_csv("lstm/y_vaild.csv").iloc[:, 1:].values
    y_test = pd.read_csv("lstm/y_test_v2.csv").iloc[:, 1:].values

    # Build and prune the model
    model = build_lightweight_model(50, 1)
    steps = int(np.ceil(len(x_train) / 32))
    pruning = {'pruning_schedule': sparsity.PolynomialDecay(0.0, 0.5, 0, steps * 5)}
    pruned = sparsity.prune_low_magnitude(model, **pruning)
    pruned.compile(Adam(1e-4), 'binary_crossentropy', ['accuracy'])

    # Train the model
    cb = [sparsity.UpdatePruningStep(), SparsityCallback(), EpochLogger((x_val, y_val), log_fp)]
    pruned.fit(x_train, y_train, epochs=1000, batch_size=32,
               validation_data=(x_val, y_val), callbacks=cb,
               verbose=2, shuffle=False)
    log_fp.close()

    # Strip pruning and save the model
    final = sparsity.strip_pruning(pruned)
    final.save("lstm_model_pruned.h5")

    # Evaluate the model on test data
    y_prob = final.predict(x_test, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int).flatten()
    y_true = y_test.flatten()
    print("\n=== Test Metrics ===")
    print(classification_report(y_true, y_pred))

    # Save Integrated Gradients
    ig = integrated_gradients(x_test[0], final, m_steps=100)
    np.savetxt("ig_sample0.csv", ig.reshape(-1, 1), delimiter=",", header="attr", comments="")
    print("Saved Integrated Gradients for sample 0 -> ig_sample0.csv")

    # Perform ablation and save the plots
    visualize_ablation(final, x_test, sample_indices=[0, 1, 2], save_dir="./ablation_plots")

if __name__ == "__main__":
    main()
