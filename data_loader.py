import torch
from torch.utils.data import Dataset
import os
import pickle

def load_data_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data


import torch
from torch.utils.data import Dataset
import os
import numpy as np


class VoiceDataset(Dataset):
    def __init__(self, pickle_files_directory, labels, transform=None):
        self.pickle_files_directory = pickle_files_directory
        self.pickle_files = [os.path.join(pickle_files_directory, f) for f in os.listdir(pickle_files_directory) if
                             f.endswith('.pkl')]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Assuming the total length is the number of samples times the number of files
        return len(self.pickle_files) * 500

    def __getitem__(self, idx):
        file_idx = idx // 500
        sample_idx = idx % 500

        data = load_data_from_pickle(self.pickle_files[file_idx])
        sample = data[sample_idx]

        label = self.labels[idx]  # Access the corresponding label directly

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is of type long for classification tasks

        if self.transform:
            sample = self.transform(sample)

        return sample, label
