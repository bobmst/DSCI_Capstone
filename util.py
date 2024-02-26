# Settings for global variables
data_address = "./data/"
DATA_ADDRESS = "./data/"

# Some useful functions
import pickle
import os
from tqdm import tqdm
import numpy as np

def get_cuda_device():
    import torch
    # Check if GPUs are available
    if torch.cuda.is_available():
        # Set the default device to GPU
        torch.cuda.set_device(0)  # Specify the GPU device index if using multiple GPUs
        device = torch.device('cuda')
    else:
        # Set the default device to CPU
        # torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')

    return device

import torch.nn.functional as F
def predict(model, data_loader, device):
    pred_list = []
    out_prob_list = []
    label_list = []

    for j, (audio, label) in tqdm(enumerate(data_loader)):
        audio = audio.to(device)
        label = label.to(device)
        output = model.forward(audio)
        # loss = criterion(output, label)

        out_prob = F.softmax(output, dim=1)

        # Get prediction with np.argmax
        pred = output.argmax(dim=1, keepdim=True)

        pred = pred.detach().cpu().numpy().reshape(-1)
        truth = label.detach().cpu().numpy().reshape(-1)

        pred_list.append(pred)
        out_prob_list.append(out_prob.detach().cpu().numpy())
        label_list.append(truth)

    return pred_list, out_prob_list, label_list

def plot_train_val_metrics(train_losses, val_losses, filename="loss.png", xlab="Epoch",
                           ylab="Loss", title="Training/Validation"):
    import matplotlib.pyplot as plt
    epoch_range = range(1, len(train_losses) + 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    # plt.xticks(epoch_range)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

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

def generate_model_path(description, parent_dir="models"):
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add description to current date and time
    current_datetime += f"_{description}"

    # Create the model directory based on the current date and time
    model_dir = f"model_{current_datetime}"

    # Create the full model path by joining the parent directory and the model directory
    model_path = os.path.join(parent_dir, model_dir)

    # Create the model directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    return model_path
