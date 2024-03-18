import pandas as pd
import numpy as np
import pickle
import os
from pprint import pprint
import re
from collections import defaultdict

import librosa
import soundfile as sf
import os
import re

import preprocessing_pipeline
import models_cnn
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from util import *


def read_signal(path, sr=None):
    signal, sr = librosa.load(path, sr=sr)
    return signal, sr


class audio_loader:

    def process_file_segmentation(self, signal, sr, segment_length=5, denoise=True):
        # Apply denoise before cutting the segments
        if denoise:
            signal = preprocessing_pipeline.reduce_noise(signal, sr)
        duration = len(signal) / sr
        ls_file = []

        if duration > segment_length:
            num_segments = int(duration // segment_length)
            for i in range(num_segments):
                start_time = i * segment_length
                end_time = (i + 1) * segment_length
                segment = signal[int(start_time * sr) : int(end_time * sr)]
                ls_file.append(segment)
            # Handle the last segment if there is a remainder
            if duration % segment_length > 0:
                start_time = num_segments * segment_length
                end_time = duration
                segment = signal[int(start_time * sr) : int(end_time * sr)]
                ls_file.append(segment)
        else:
            # If the file is shorter than or equal to the segment_length, put it in the list
            ls_file.append(signal)
        return ls_file

    def process_raw(
        self, segment_length=5, denoise=True, n_fft=2048, n_mels=128, n_mfcc=20
    ):
        ls_new_signal = []
        # Cut audio into segments with given length
        for audio in self.ls_raw_signal:
            ls_new_signal += self.process_file_segmentation(
                signal=audio, sr=self.sr, segment_length=segment_length, denoise=denoise
            )
        ls_mfcc = []
        for new_signal in ls_new_signal:
            _, _, mfcc = preprocessing_pipeline.process_audio(
                new_signal,
                self.sr,
                length=segment_length,
                n_fft=n_fft,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                denoise=False,
            )
            # print(mfcc.shape)
            ls_mfcc.append(mfcc)
        self.preprocessed = np.array(ls_mfcc)
        print("Preprocess finished: ", self.preprocessed.shape)

    def __init__(self, ls_raw_signal: list, sr):
        # list of raw audio data
        self.ls_raw_signal = ls_raw_signal
        self.sr = sr
        self.preprocessed = None


class auth_model:
    def train(
        self, train_audio_loader, n_epochs, patience, model_save_path, learning_rate
    ):
        X_false = self.X_false
        y_false = np.zeros(len(X_false), dtype=int)
        X_true = train_audio_loader.preprocessed
        y_true = np.ones(len(X_true), dtype=int)
        X_train = np.concatenate((X_false, X_true), axis=0)
        y_train = np.concatenate((y_false, y_true), axis=0)
        X_val = X_true[:1]
        y_val = y_true[:1]

        train_dataset = models_cnn.VoiceDataset(X_train, y_train)
        val_dataset = models_cnn.VoiceDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Freeze all layers first
        for param in self.clf.parameters():
            param.requires_grad = False

        # Unfreeze the last two fully connected layers
        for param in self.clf.fc1.parameters():
            param.requires_grad = True
        for param in self.clf.fc2.parameters():
            param.requires_grad = True

        self.clf.fc2 = nn.Linear(in_features=128, out_features=1, bias=True)
        self.clf = self.clf.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=learning_rate)

        self.clf.fit_binary(
            train_loader,
            val_loader,
            criterion,
            optimizer,
            self.device,
            n_epochs,
            patience,
            model_save_path,
        )
        print("Finished training")
        return "Finished training"

    def predict(self, speaker_audio_loader, batch_size, accept_threshold):
        self.clf = self.clf.to(self.device)
        speaker_X = speaker_audio_loader.preprocessed
        speaker_y = np.ones(len(speaker_X), dtype=int)
        speaker_dataset = models_cnn.VoiceDataset(speaker_X, speaker_y)
        speaker_loader = DataLoader(
            speaker_dataset, batch_size=self.batch_size, shuffle=True
        )
        pred_list, prob_list, label_list = self.clf.predict_binary(
            speaker_loader, self.device, threshold=accept_threshold
        )

        return pred_list, prob_list

        # need to change the predict here

    def __init__(self, clf, X_false, batch_size):
        self.device = get_cuda_device()
        self.clf = clf
        self.X_false = X_false
        self.batch_size = batch_size
