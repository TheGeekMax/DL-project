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
    inputs = Input(shape=(input_shape,))

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


def create_cnn_model(input_shape, dropout_rate=0.5, l2_reg=0.001):
    """
    Crée un modèle Convolutional Neural Network (CNN) avec l'API fonctionnelle de Keras.

    Args:
        input_shape: Forme des données d'entrée (nombre de points temporels, canaux)
        dropout_rate: Taux de dropout pour la régularisation
        l2_reg: Facteur de régularisation L2

    Returns:
        model: Modèle CNN compilé
    """
    # Couche d'entrée
    inputs = Input(shape=input_shape)

    # Premier bloc convolutionnel
    x = Conv1D(
        filters=64,
        kernel_size=8,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_reg),
    )(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)

    # Deuxième bloc convolutionnel
    x = Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)

    # Troisième bloc convolutionnel
    x = Conv1D(
        filters=256,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    # Couche dense
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Couche de sortie (classification binaire)
    outputs = Dense(1, activation="sigmoid")(x)

    # Création du modèle
    model = Model(inputs=inputs, outputs=outputs, name="CNN_Model")

    return model


def create_rnn_model(input_shape, rnn_type="lstm", dropout_rate=0.5, l2_reg=0.001):
    """
    Crée un modèle Recurrent Neural Network (RNN) avec l'API fonctionnelle de Keras.

    Args:
        input_shape: Forme des données d'entrée (nombre de points temporels, features)
        rnn_type: Type de RNN ('lstm' ou 'gru')
        dropout_rate: Taux de dropout pour la régularisation
        l2_reg: Facteur de régularisation L2

    Returns:
        model: Modèle RNN compilé
    """
    # Couche d'entrée
    inputs = Input(shape=input_shape)

    # Première couche récurrente bidirectionnelle
    if rnn_type.lower() == "lstm":
        x = Bidirectional(
            LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg))
        )(inputs)
    else:  # GRU
        x = Bidirectional(
            GRU(64, return_sequences=True, kernel_regularizer=l2(l2_reg))
        )(inputs)

    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Deuxième couche récurrente
    if rnn_type.lower() == "lstm":
        x = LSTM(128, kernel_regularizer=l2(l2_reg))(x)
    else:  # GRU
        x = GRU(128, kernel_regularizer=l2(l2_reg))(x)

    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Couche dense
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Couche de sortie (classification binaire)
    outputs = Dense(1, activation="sigmoid")(x)

    # Création du modèle
    model_name = "LSTM_Model" if rnn_type.lower() == "lstm" else "GRU_Model"
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def compile_model(model, optimizer="adam", learning_rate=0.001):
    """
    Compile le modèle avec les paramètres spécifiés.

    Args:
        model: Modèle Keras à compiler
        optimizer: Optimiseur à utiliser
        learning_rate: Taux d'apprentissage

    Returns:
        model: Modèle compilé
    """
    if optimizer.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def count_parameters(model):
    """
    Compte le nombre de paramètres dans un modèle.

    Args:
        model: Modèle Keras

    Returns:
        total_params: Nombre total de paramètres
        trainable_params: Nombre de paramètres entraînables
        non_trainable_params: Nombre de paramètres non entraînables
    """
    import numpy as np

    # Version corrigée utilisant .shape au lieu de .get_shape()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_weights)
    non_trainable_params = sum(np.prod(v.shape) for v in model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params

    return total_params, trainable_params, non_trainable_params
