## eda.py
import numpy as np
import librosa
from util import data_address
from preprocessing_pipeline import *

FILE_DIR = data_address+"VOiCES_devkit/"

def descriptive_statistics(arr):
    # n_min = arr.min()
    # n_max = arr.max()
    n_mean = arr.mean()
    n_std = arr.std()

    return (n_mean, n_std)

def map_columns_statistics(row,spectrogram,column_name):
    for i in range(spectrogram.shape[0]):
        n_mean, n_std = descriptive_statistics(spectrogram[i])
        # row[column_name+'_'+str(i)+'_min'] = n_min
        # row[column_name+'_'+str(i)+'_max'] = n_max
        row[column_name+'_'+str(i)+'_mean'] = n_mean
        row[column_name+'_'+str(i)+'_std'] = n_std

    return row

def apply_process_signal(row, n_mels=128, n_mfcc=20):
    filename = row['filename']
    signal, sr = librosa.load(FILE_DIR+filename)
    _,melspectrogram,mfcc = process_audio(signal,sr, n_mels, n_mfcc)
    
    row = map_columns_statistics(row,melspectrogram,'mel')
    row = map_columns_statistics(row,mfcc,'mfcc')
        
    return row

    