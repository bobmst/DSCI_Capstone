import os, logging, sys

# Settings for global variables
# data_address = "./data/"
DATA_ADDRESS = "./data/"
TRAIN_DIR = os.path.join(DATA_ADDRESS, "preprocessed", "train")
TEST_DIR = os.path.join(DATA_ADDRESS, "preprocessed", "test")

# Some useful functions
import pickle, re
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger("util_logger")


def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


## decrypted
# def load_train(train_dir, train_files):
#     """_summary_

#     Args:
#         train_dir (string): Path to the folder with the processed training data
#         train_files (list): List of train files read from os, and filtered with desired prefix

#     Returns:
#         numpy.array: 3D array with all the training data. Dim:(n_train_data, n_features, n_data_length)
#     """
#     ls_X_trian = []
#     for train_file in tqdm(train_files):
#         file_dir = os.path.join(train_dir, train_file)
#         ls_X_trian.append(load_pickle(file_dir))
#     return np.concatenate(ls_X_trian, axis=0)


def load_feature_data(dir_feature, feature_files):
    ls_X = []
    for k, v in tqdm(feature_files.items()):
        file = v["file_name"]
        file_dir = os.path.join(dir_feature, file)
        ls_X.append(load_pickle(file_dir))
    return np.concatenate(ls_X, axis=0)


# can be rewrite into oop data loader, with X, y, metadata
def load_data(dir_feature, file_prefix, dir_df_index, n_interval=500, flatten=False):
    # load X
    files = os.listdir(dir_feature)
    files_mfcc = [file for file in files if file.startswith(file_prefix)]
    pattern = re.compile(r"_(\d+)-(\d+)\.pkl$")
    dir_files = defaultdict(dict)
    for file_name in files_mfcc:
        match = pattern.search(file_name)
        n1 = int(match.group(1))
        n2 = int(match.group(2))
        file_num = n1 / n_interval
        dir_files[file_num] = {
            "begin": n1,
            "end": n2,
            "file_name": file_name,
        }
    sorted_dir_files = {k: dir_files[k] for k in sorted(dir_files)}
    X = load_feature_data(dir_feature=dir_feature, feature_files=sorted_dir_files)
    logger.debug(f"X Shape:{X.shape}")

    # load Y
    df_index = pd.read_pickle(dir_df_index)
    y = np.array(df_index["speaker"]).astype("float32")
    logger.debug(f"y Shape:{y.shape}")

    if flatten:
        X = X.reshape(X.shape[0], -1)
        logger.debug(f"X Shape after flattened:{X.shape}")

        X_df = pd.DataFrame(X)
        y_series = pd.Series(y, name="target")
        logger.info(f"y has been assigned to column 'target'")
        data = pd.concat([X_df, y_series], axis=1)
        logger.debug(f"final dataframe shape{data.shape}")
        print(f"Loaded dataframe\n with flattend X\n with shape: {data.shape}")
        return data
    else:
        print(f"Loaded np array\n with X shape: {X.shape}\n with y shape: {y.shape}")
        return X, y


def get_cuda_device():
    import torch

    # Check if GPUs are available
    if torch.cuda.is_available():
        # Set the default device to GPU
        torch.cuda.set_device(0)  # Specify the GPU device index if using multiple GPUs
        device = torch.device("cuda")
    else:
        # Set the default device to CPU
        # torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device("cpu")

    return device
