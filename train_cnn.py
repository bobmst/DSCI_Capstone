
"""
This script prepares the processes audio files for training and testing models.

Usage:
    train_cnn.py --train_dir TD --index_path IP --output_dir OD --params P
    train_cnn.py -h | --help
Options:
    --train_dir
        path to preprocessed audio files for training
    --index_path
        path to index file for training
    --output_dir
        path to output directory to save results (if does not exists, it will be created)
    --params
        path to parameters yaml file
"""

from docopt import docopt
import os
from pathlib import Path
import importlib
import logging
import yaml

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from util import *
from data_prepare import DataPrep
from models_cnn import VoiceDataset


_log = logging.getLogger("train_cnn")

def get_args(args):

    if args["--train_dir"]:
        train_dir = args["TD"]
    else:
        raise RuntimeError("Please specify train directory")

    if args["--index_path"]:
        index_path = args["IP"]
    else:
        raise RuntimeError("Please specify index path")

    if args["--output_dir"]:
        output_dir = args["OD"]
    else:
        raise RuntimeError("Please specify output directory")

    if args["--params"]:
        params_path = args["P"]
    else:
        raise RuntimeError("Please specify parameters file path")

    return train_dir, index_path, output_dir, params_path

def get_model(algo):
    NAME = list(algo["name"].keys())[0]
    module = importlib.import_module("models_cnn")

    if "num_classes" in algo["name"][NAME]:
        model = getattr(module, NAME)(**algo["name"][NAME])

    return model

def main(args):
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.INFO)

    train_dir, index_path, output_dir, params_path = get_args(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)

    algo = params["algorithm"]
    batch_size = int(algo["params"]["batch_size"])
    n_epochs = int(algo["params"]["epochs"])
    patience = int(algo["params"]["patience"])
    learning_rate = float(algo["params"]["learning_rate"])

    model_save_path = output_dir + '/'+ algo["save_path"]

    # Get data
    data = DataPrep(train_dir, index_path)
    data.load()
    data.get_data_info()
    X, y, labels = data.get_data(labels=True)

    # Prepare data for training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create VoiceDataset object
    train_dataset = VoiceDataset(X_train, y_train)
    val_dataset = VoiceDataset(X_val, y_val)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = get_cuda_device()
    model = get_model(algo)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    _log.info("Training model...")

    model.fit(train_loader,
              val_loader,
              criterion,
              optimizer,
              device,
              n_epochs,
              patience,
              model_save_path)

    _log.info(f"Model saved at {model_save_path}")

    # Plot metrics
    plot_train_val_metrics(model.train_losses,
                           model.val_losses,
                           filename=output_dir + '/loss.png',
                           xlab="Epoch",
                           ylab="Loss",
                           title="Training/Validation Loss")

    plot_train_val_metrics(model.train_accuracies,
                              model.val_accuracies,
                              filename=output_dir + '/accuracy.png',
                              xlab="Epoch",
                              ylab="Accuracy",
                              title="Training/Validation Accuracy")

    # Predict
    pred_list, label_list = model.predict(val_loader, device)

    # Save predictions and labels
    save_pickle(output_dir + '/predictions.pkl', (pred_list, label_list))

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)



