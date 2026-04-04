import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity import keras as sparsity

def attention_block(inputs):
    feature_dim = inputs.shape[-1]
    att_probs = layers.Dense(feature_dim, activation='softmax')(inputs)
    return layers.Multiply()([inputs, att_probs])

def build_model(n_steps, n_features):
    inp = Input(shape=(n_steps, n_features))

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
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prune_model(model, train_len, batch=32, epochs=500):
    steps = int(tf.ceil(train_len / batch))
    pruning = sparsity.PolynomialDecay(0.0, 0.5, 0, steps*epochs)
    pruned = sparsity.prune_low_magnitude(model, pruning_schedule=pruning)
    pruned.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return pruned

def strip(pruned):
    return sparsity.strip_pruning(pruned)

def pruning_callback():
    return [sparsity.UpdatePruningStep()]
