import tensorflow as tf
from tensorflow.keras import layers, Input, Model, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np

def attention_block(inputs):
    feature_dim = inputs.shape[-1]
    att_weights = layers.Dense(feature_dim, activation='softmax')(inputs)
    att_output = layers.Multiply()([inputs, att_weights])
    return att_output

def build_arclight_attgru(input_shape=(50, 1)):
    inp = Input(shape=input_shape)

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
    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.Precision(), metrics.Recall()]
    )
    return model

def build_pruned_arclight_attgru(input_shape=(50, 1), final_sparsity=0.5):
    base_model = build_arclight_attgru(input_shape=input_shape)
    pruning_schedule = sparsity.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=500 * int(np.ceil(1000 / 32))
    )
    pruned_model = sparsity.prune_low_magnitude(base_model, pruning_schedule=pruning_schedule)
    pruned_model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.Precision(), metrics.Recall()]
    )
    return pruned_model

if __name__ == "__main__":
    model = build_arclight_attgru()
    model.summary()
