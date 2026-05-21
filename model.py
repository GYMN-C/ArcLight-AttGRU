import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam


def attention_block(inputs):
    feature_dim = inputs.shape[-1]
    att_probs = layers.Dense(feature_dim, activation='softmax')(inputs)
    output = layers.Multiply()([inputs, att_probs])
    return output


def build_model(
    n_steps,
    n_features,
    conv_filters=32,
    gru_units=32,
    dense1_units=64,
    dense2_units=32
):
    inp = Input(shape=(n_steps, n_features))

    x = layers.SeparableConv1D(
        filters=conv_filters,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(inp)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv1D(
        filters=conv_filters,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.GRU(
        units=gru_units,
        return_sequences=True
    )(x)

    x = layers.GRU(
        units=gru_units,
        return_sequences=True
    )(x)

    x = attention_block(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        units=dense1_units,
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        units=dense2_units,
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(
        units=1,
        activation='sigmoid'
    )(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_structured_pruned_model(n_steps, n_features):
    model = build_model(
        n_steps=n_steps,
        n_features=n_features,
        conv_filters=32,
        gru_units=32,
        dense1_units=64,
        dense2_units=32
    )

    return model


if __name__ == "__main__":
    n_steps = 100
    n_features = 10

    model = build_structured_pruned_model(
        n_steps=n_steps,
        n_features=n_features
    )

    model.summary()
