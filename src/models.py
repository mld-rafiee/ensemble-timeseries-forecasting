"""Deep learning model definitions: RNN, LSTM, GRU, Transformer."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
    SimpleRNN,
)


def positional_encoding(n_steps, d_model):
    """Generate positional encoding for Transformer."""
    positions = np.arange(n_steps)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_model(model_type, n_steps, n_features, n_future_steps):
    """
    Create a deep learning model.

    Args:
        model_type: 'RNN', 'LSTM', 'GRU', or 'Transformer'
        n_steps: Number of past time steps (window size)
        n_features: Number of features
        n_future_steps: Number of future steps to predict
    """
    output_units = n_future_steps * n_features

    if model_type == "Transformer":
        d_model = 64
        n_heads = 8
        ff_units = 512
        dropout_rate = 0.1

        inputs = Input(shape=(n_steps, n_features))

        # Positional encoding
        pos_encoding = positional_encoding(n_steps, d_model)
        x = Dense(d_model)(inputs) + pos_encoding[:, :n_steps, :]

        # Multi-Head Attention
        attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        attn_output = LayerNormalization(epsilon=1e-6)(attn_output + x)

        # Feedforward
        ff_output = Dense(ff_units, activation="relu")(attn_output)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = LayerNormalization(epsilon=1e-6)(ff_output + attn_output)

        # Output
        outputs = Dense(output_units)(ff_output[:, -1])
        outputs = Reshape((n_future_steps, n_features))(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mae",
            metrics=["mse"],
        )

    else:
        # RNN, LSTM, or GRU
        model = Sequential()

        if model_type == "RNN":
            model.add(
                SimpleRNN(
                    units=64, input_shape=(n_steps, n_features), activation="relu"
                )
            )
        elif model_type == "LSTM":
            model.add(
                LSTM(units=64, input_shape=(n_steps, n_features), activation="relu")
            )
        elif model_type == "GRU":
            model.add(
                GRU(units=64, input_shape=(n_steps, n_features), activation="relu")
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.add(Dense(units=output_units))
        model.add(Reshape((n_future_steps, n_features)))
        model.compile(
            loss="mae",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["mse"],
        )

    return model
