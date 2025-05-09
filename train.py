import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import requests as req
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils import create_if_non_existant, download_dataset_if_non_existant
from src.graph_gen import generate_graphs_from_model
from tensorflow.keras.utils import plot_model

# train sur ce modèle
# sauvegarde dans le dossier models
from src.utils import get_args_float


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
        --RNN-cla -> create a RNN model (clasic)
        --RNN-bidir -> create a RNN model (bidirectional)
        --mlp -> create an MLP model
        --dr X -> set dropout rate to X (for MLP only)
        --l2_reg Y -> set L2 regularization to Y (for MLP only)
    """
    args = sys.argv[1:]
    model_type = None
    mlp_dropout_rate = 0.5
    mlp_l2_reg = 0.001

    if "--cnn" in args:
        model_type = "CNN"
    elif "--rnn-cla" in args:
        model_type = "RNN-clasic"
    elif "--rnn-bidir" in args:
        model_type = "RNN-bidirectional"
    elif "--mlp" in args:
        model_type = "mlp"
        mlp_dropout_rate = get_args_float(args, "--dr", default=0.5)
        mlp_l2_reg = get_args_float(args, "--l2_reg", default=0.001)
        print(f"Dropout rate : {mlp_dropout_rate}")
        print(f"L2 regularization : {mlp_l2_reg}")
    else:
        print("Please specify a model type : --cnn, --rnn-cla, --rnn-bidirectional, --mlp")
        sys.exit(1)

    
    
    if mlp_dropout_rate < 0 or mlp_dropout_rate > 1:
        print("ERROR : Dropout rate must be between 0 and 1")
        sys.exit(1)
    if mlp_l2_reg < 0:
        print("ERROR : L2 regularization must be greater than 0")
        sys.exit(1)
    
    

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

    # step 2.1.1 : sort to have same count of 1 and 0 in the train set
    train = (train_x, train_y)
    train_0 = train[0][train[1] == 0]
    train_1 = train[0][train[1] == 1]

    # Find the smaller class size to ensure balance
    count_0 = train_0.shape[0]
    count_1 = train_1.shape[0]
    balanced_count = min(count_0, count_1)

    # Take equal numbers from each class
    train_0 = train_0[:balanced_count]
    train_1 = train_1[:balanced_count]

    # Combine balanced datasets
    train_x = np.concatenate((train_0, train_1), axis=0)
    train_y = np.concatenate((np.zeros(balanced_count), np.ones(balanced_count)), axis=0)

    # Shuffle the combined data
    indices = np.arange(len(train_y))
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

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
    elif model_type == "RNN-clasic":
        from src.models import create_rnn_model
        model = create_rnn_model(input_shape=train_x.shape[1:], rnn_type="clasic")
    elif model_type == "RNN-bidirectional":
        from src.models import create_rnn_model
        model = create_rnn_model(input_shape=train_x.shape[1:], rnn_type="bidirectional")
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
    hystory = model.fit(train_x, train_y, epochs=100, batch_size=256, verbose=1, validation_data=(test_x, test_y))

    # step 5.1 : generate the graphs
    y_pred = model.predict(test_x)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    generate_graphs_from_model(hystory, model_type, test_x, test_y, y_pred)

    # step 5.2 : generate graph of different layers
    # using plot_model
    plot_model(model, to_file=f"graphs/{model_type}_model.png", show_shapes=True, show_layer_names=True)

    # step 6 : save the model
    model.save(f"models/{model_type}.h5")
    print(f"Model saved in models/{model_type}.h5")

