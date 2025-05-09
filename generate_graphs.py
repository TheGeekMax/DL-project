import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.utils import create_if_non_existant, download_dataset_if_non_existant
from src.graph_gen import generate_graphs_from_model

def get_model_from_name(name:str):
    #get from de models folder
    model_path = os.path.join("models", name+".h5")
    if not os.path.exists(model_path):
        print(f"Model {name} does not exist")
        sys.exit(1)
    model = load_model(model_path)
    return model

if __name__ == "__main__":
    model_name = sys.argv[1]
    model = get_model_from_name(model_name)

    # step 0 : create all the folders
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

    # step 3 : predict the data
    train_y_pred = model.predict(train_x)
    test_y_pred = model.predict(test_x)

    test_y_pred = np.where(test_y_pred > 0.5, 1, 0)
    train_y_pred = np.where(train_y_pred > 0.5, 1, 0)

    # step 4 : generate the graphs
    generate_graphs_from_model(None, model_name, test_x, test_y, test_y_pred)