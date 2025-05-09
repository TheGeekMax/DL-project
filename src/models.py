import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    LSTM,
    GRU,
    Bidirectional,
    Flatten,
    Activation,
)
from tensorflow.keras.regularizers import l2
import tensorflow

import numpy as np


def create_mlp_model(input_shape, dropout_rate=0.5, l2_reg=0.001):
    """
    Crée un modèle Multi-Layer Perceptron (MLP) avec l'API fonctionnelle de Keras.

    Args:
        input_shape: Forme des données d'entrée (nombre de points temporels)
        dropout_rate: Taux de dropout pour la régularisation
        l2_reg: Facteur de régularisation L2

    Returns:
        model: Modèle MLP compilé
    """
    # Couche d'entrée
    inputs = Input(shape=input_shape)

    # Première couche cachée
    x = Dense(256, activation="relu", kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Deuxième couche cachée
    x = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Troisième couche cachée
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Couche de sortie (classification binaire)
    outputs = Dense(1, activation="sigmoid")(x)

    # Création du modèle
    model = Model(inputs=inputs, outputs=outputs, name="MLP_Model")

    return model


def create_cnn_model(input_shape, dropout_rate=0.1, l2_penalty=0.005):
    """
    Construction d'un modèle CNN adapté avec Flatten et nommage structuré des couches.

    Args:
        input_shape: Forme des entrées (timesteps, channels)
        dropout_rate: Taux de dropout
        l2_penalty: Coefficient de régularisation L2

    Returns:
        model: Modèle CNN compilé
    """

    input_layer = Input(shape=input_shape)

    # Bloc 1
    conv_layer_1 = Conv1D(128, kernel_size=7, padding='same', kernel_regularizer=l2(l2_penalty))(input_layer)
    batchnorm_layer_1 = BatchNormalization()(conv_layer_1)
    activation_layer_1 = Activation('relu')(batchnorm_layer_1)
    pooling_layer_1 = MaxPooling1D(pool_size=2)(activation_layer_1)
    dropout_layer_1 = Dropout(rate=dropout_rate * 0.5)(pooling_layer_1)

    # Bloc 2
    conv_layer_2 = Conv1D(256, kernel_size=5, padding='same', kernel_regularizer=l2(l2_penalty))(dropout_layer_1)
    batchnorm_layer_2 = BatchNormalization()(conv_layer_2)
    activation_layer_2 = Activation('relu')(batchnorm_layer_2)
    pooling_layer_2 = MaxPooling1D(pool_size=2)(activation_layer_2)
    dropout_layer_2 = Dropout(rate=dropout_rate * 0.6)(pooling_layer_2)

    # Bloc 3
    conv_layer_3 = Conv1D(512, kernel_size=3, padding='same', kernel_regularizer=l2(l2_penalty))(dropout_layer_2)
    batchnorm_layer_3 = BatchNormalization()(conv_layer_3)
    activation_layer_3 = Activation('relu')(batchnorm_layer_3)
    pooling_layer_3 = MaxPooling1D(pool_size=2)(activation_layer_3)
    dropout_layer_3 = Dropout(rate=dropout_rate * 0.7)(pooling_layer_3)

    # Flatten + Dense
    flatten_layer = Flatten()(dropout_layer_3)
    dense_layer_1 = Dense(256, activation='relu', kernel_regularizer=l2(l2_penalty))(flatten_layer)
    dropout_layer_4 = Dropout(rate=dropout_rate)(dense_layer_1)

    output_layer = Dense(1, activation='sigmoid')(dropout_layer_4)

    model = Model(inputs=input_layer, outputs=output_layer, name="Custom_CNN_Flatten")

    return model


def create_rnn_model(input_shape, rnn_type="clasic", dropout_rate=0.7 , l2_reg=0.001):
    inputs = Input(shape=input_shape)

    # Première couche récurrente bidirectionnelle
    if rnn_type.lower() == "clasic":
        lstm_layer1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg))(inputs)
        dropout_layer1 = Dropout(dropout_rate)(lstm_layer1)
        lstm_layer2 = LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_reg))(dropout_layer1)
        dropout_layer2 = Dropout(dropout_rate)(lstm_layer2)
    else:
        lstm_layer1 = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(inputs)
        dropout_layer1 = Dropout(dropout_rate)(lstm_layer1)
        lstm_layer2 = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_reg)))(dropout_layer1)
        dropout_layer2 = Dropout(dropout_rate)(lstm_layer2)

    outputs = Dense(1, activation="sigmoid")(dropout_layer2)

    # Création du modèle
    model_name = "RNN_Clasic" if rnn_type.lower() == "clasic" else "RNN_Bidirectional"
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model
