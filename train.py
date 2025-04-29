import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# train sur ce modèle
# sauvegarde dans le dossier models

def create_if_non_existant(path : str):
    if not os.path.exists(path):
        os.makedirs(path)

def download_dataset_if_non_existant(url : str, path : str):
    if not os.path.exists(path):
        os.system(f"wget {url} -O {path}")


if __name__ == "__main__":
    ## step -1 : get the arguments
    """
        --CNN -> create a CNN model
        --RNN-LSTM -> create a RNN model (LSTM)
        --RNN-GRU -> create a RNN model (GRU)
        --mlp -> create an MLP model
        --dr X -> set dropout rate to X (for MLP only)
        --l2_reg Y -> set L2 regularization to Y (for MLP only)
    """
    args = sys.argv[1:]
    model_type = None
    mlp_dropout_rate = 0.5
    mlp_l2_reg = 0.001

    if "--CNN" in args:
        model_type = "CNN"
    elif "--RNN-LSTM" in args:
        model_type = "RNN-LSTM"
    elif "--RNN-GRU" in args:
        model_type = "RNN-GRU"
    elif "--mlp" in args:
        model_type = "mlp"
    else:
        print("Please specify a model type : --CNN, --RNN-LSTM, --RNN-GRU, --mlp")
        sys.exit(1)

    def get_args(arg_name: str, default: str):
        if arg_name in args:
            index = args.index(arg_name)
            if index + 1 < len(args):
                return args[index + 1]
        return default

    mlp_dropout_rate = get_args("--dr", default=0.5)
    mlp_l2_reg = get_args("--l2_reg", default=0.001)
    try:
        mlp_dropout_rate = float(mlp_dropout_rate)
        mlp_l2_reg = float(mlp_l2_reg)
    except ValueError:
        print("ERROR : Dropout rate and L2 regularization must be float values")
        sys.exit(1)
    if mlp_dropout_rate < 0 or mlp_dropout_rate > 1:
        print("ERROR : Dropout rate must be between 0 and 1")
        sys.exit(1)
    if mlp_l2_reg < 0:
        print("ERROR : L2 regularization must be greater than 0")
        sys.exit(1)
    print(f"Model type : {model_type}")
    print(f"Dropout rate : {mlp_dropout_rate}")
    print(f"L2 regularization : {mlp_l2_reg}")

    # step 0 : create all the folders
    create_if_non_existant("models")
    create_if_non_existant("datasets")
    create_if_non_existant("graphs")

    # step 1 : load the datasets
    download_dataset_if_non_existant("https://maxime-devanne.com/datasets/Earthquakes/Earthquakes_TRAIN.tsv", "datasets/train.tsv")
    download_dataset_if_non_existant("https://maxime-devanne.com/datasets/Earthquakes/Earthquakes_TEST.tsv", "datasets/test.tsv")

    # step 2 : get the data
    train_df = pd.read_csv("datasets/train.tsv", sep="\t")
    test_df = pd.read_csv("datasets/test.tsv", sep="\t")

    # step 2.1 : preprocess the data
    train_x = train_df.iloc[:,1:].values
    train_y = train_df.iloc[:,0].values

    test_x = test_df.iloc[:,1:].values
    test_y = test_df.iloc[:,0].values

    # step 2.2 : reshape the data
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    # step 2.3 : normalize the data
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_x)

    train_x = normalizer(train_x)
    test_x = normalizer(test_x)

    # step 3 : create the model
    model = None
    if model_type == "CNN":
        from src.models import create_cnn_model
        model = create_cnn_model(input_shape=train_x.shape[1:])
    elif model_type == "RNN-LSTM":
        from src.models import create_rnn_model
        model = create_rnn_model(input_shape=train_x.shape[1:], rnn_type="lstm")
    elif model_type == "RNN-GRU":
        from src.models import create_rnn_model
        model = create_rnn_model(input_shape=train_x.shape[1:], rnn_type="gru")
    elif model_type == "mlp":
        from src.models import create_mlp_model
        model = create_mlp_model(
            input_shape=train_x.shape[1:],
            dropout_rate=mlp_dropout_rate,
            l2_reg=mlp_l2_reg
        )
    else:
        print("ERROR : Model type not found")
        sys.exit(1)
    
    # step 4 : compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    # step 5 : train the model
    hystory = model.fit(train_x, train_y, epochs=1000, batch_size=256, validation_split=0.2, verbose=1)

    # step 5.1 : generate the graphs
    plt.plot(hystory.history["loss"], label="loss")
    plt.plot(hystory.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"graphs/{model_type}_loss.png")

    plt.clf()
    plt.plot(hystory.history["accuracy"], label="accuracy")
    plt.plot(hystory.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"graphs/{model_type}_accuracy.png")

    plt.clf()
    plt.plot(hystory.history["loss"], label="loss")
    plt.plot(hystory.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"graphs/{model_type}_loss.png")
    
    plt.clf()
    plt.plot(hystory.history["accuracy"], label="accuracy")
    plt.plot(hystory.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"graphs/{model_type}_accuracy.png")

    # step 6 : save the model
    model.save(f"models/{model_type}.h5")
    print(f"Model saved in models/{model_type}.h5")

