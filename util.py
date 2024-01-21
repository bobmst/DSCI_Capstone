# Settings for global variables
data_address = "./data/"
DATA_ADDRESS = "./data/"


# Some useful functions
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np


def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_train(train_dir, train_files):
    """_summary_

    Args:
        train_dir (string): Path to the folder with the processed training data
        train_files (list): List of train files read from os, and filtered with desired prefix

    Returns:
        numpy.array: 3D array with all the training data. Dim:(n_train_data, n_features, n_data_length)
    """
    ls_X_trian = []
    for train_file in tqdm(train_files):
        file_dir = os.path.join(train_dir, train_file)
        ls_X_trian.append(load_pickle(file_dir))
    return np.concatenate(ls_X_trian, axis=0)
