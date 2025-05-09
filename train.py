import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests as req
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils import create_if_non_existant, download_dataset_if_non_existant
from src.graph_gen import generate_graphs_from_model
from tensorflow.keras.utils import plot_model


if __name__ == "__main__":
    ## step -1 : get the arguments
    """
        --CNN -> create a CNN model
        --RNN-cla -> create a RNN model (clasic)
        --RNN-bidir -> create a RNN model (bidirectional)
        --MLP -> create a MLP model
    """
    args = sys.argv[1:]
    model_type = None
    if "--CNN" in args:
        model_type = "CNN"
    elif "--RNN-cla" in args:
        model_type = "RNN-clasic"
    elif "--RNN-bidir" in args:
        model_type = "RNN-bidirectional"
    elif "--MLP" in args:
        model_type = "MLP"
    else:
        print("Please specify a model type : --CNN, --RNN-cla, --RNN-bidir, --MLP")
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
    elif model_type == "MLP":
        from src.models import create_mlp_model
        model = create_mlp_model(input_shape=train_x.shape[1:])
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

