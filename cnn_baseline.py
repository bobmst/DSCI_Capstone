import numpy as np
import os
from util import *
import pandas as pd
from pathlib import Path

FILE_DIR = Path(DATA_ADDRESS) / "VOiCES_devkit"
TRAIN_DIR = Path(DATA_ADDRESS) / "preprocessed/train"
INDEX_DIR = Path(DATA_ADDRESS) / "preprocessed"

def prepare_index_data(path, num_classes=None):
    """
    Prepare the index data
    :param path
    :param num_classes (optional): specify an integer to add a column to subset the data to randomly selected N
    speakers (speakerN (0/1), speakerN_class (0-N-1)
    :return: 2 dataframes (df_index_reference, df_files_to_load)
    """
    df_index = pd.read_pickle(path)

    # create a column in train for count of speakers repeats
    df_index = df_index.merge(df_index.groupby('speaker').size().reset_index
                                          (name='n_samples_per_speaker'), on='speaker', how='left')

    # select unique speaker-pkl_filename
    df_index['pkl_filename'] = df_index['pkl_file_dir'].apply(lambda x: Path(x).name)
    df_index_unique = df_index[["file_index", 'speaker', 'pkl_filename', 'n_samples_per_speaker']].drop_duplicates()

    # make sure the pkl filenames are loaded in the same order as in df_index_train
    df_files_to_load = df_index_unique[["pkl_filename"]].drop_duplicates()
    df_files_to_load = df_files_to_load.reset_index().sort_values(by='index').reset_index(drop=True)

    df_index_reference = df_index_unique.merge(df_files_to_load, on='pkl_filename',
                                                           how='left').sort_values \
        (by=['index', 'file_index']) \
        .reset_index(drop=True)

    df_index_reference["speaker_full_array_index"] = np.array(range(len(df_index_reference)))
    df_index_reference["pkl_filename_index"] = df_index_reference["index"]
    df_index_reference = df_index_reference[['file_index', 'pkl_filename', 'pkl_filename_index',
                                             'n_samples_per_speaker', 'speaker',
                                             'speaker_full_array_index']]

    # prepare labels
    speaker_map = {speaker: i for i, speaker in enumerate(df_index_reference["speaker"].unique())}
    df_index_reference['speaker_class'] = df_index_reference['speaker'].map(speaker_map)

    if num_classes:
        speakers = df_index_reference[df_index_reference['n_samples_per_speaker'] == 256]['speaker'].sample(
            num_classes, random_state=2024).values

        df_index_reference[f"speaker{num_classes}"] = df_index_reference['speaker'].apply(
            lambda x: 1 if x in speakers else 0)

        speaker_map = {speaker: i for i, speaker in enumerate(speakers)}
        df_index_reference[f"speaker{num_classes}_class"] = df_index_reference['speaker'].map(speaker_map)

    return df_index_reference, df_files_to_load

def load_data(df_index_reference, df_files_to_load, input_dir, num_classes=None, verbose=False):
    """
    Load the data
    :param df_index_reference: output of prepare_index_data
    :param df_files_to_load: output of prepare_index_data
    :param input_dir: directory to pkl files location
    :param num_classes: specify an integer to subset the data to randomly selected N speakers in prepare_index_data (
    must be the same as num_classes in prepare_index_data)
    :param verbose: print the filename being loaded
    :return: X, y, labels (speaker IDs)
    """
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    X_ls = []
    filenames = []
    for index, row in tqdm(df_files_to_load.iterrows()):
        filename = row['pkl_filename']
        if verbose:
            print("Loading: ", filename)
        x = load_pickle(input_dir / filename)
        X_ls.append(x)
        filenames.append(filename)
    X = np.concatenate(X_ls, axis=0)

    labels = df_index_reference["speaker"].values
    y = df_index_reference["speaker_class"].values

    if num_classes:
        # subset to speaker15=1 and make sure it's sorted by how files will be loaded
        df_index_reference_sub = df_index_reference[df_index_reference[f"speaker{num_classes}"] == 1].sort_values \
            (by=["pkl_filename_index", "pkl_filename"]).copy()

        X = X[df_index_reference_sub["speaker_full_array_index"].values]

        labels = df_index_reference_sub["speaker"].values
        y = df_index_reference_sub[f"speaker{num_classes}_class"].values
    return X, y, labels

def main():
    df_index_reference, df_files_to_load = prepare_index_data(INDEX_DIR / "df_index_train.pkl", 15)
    X, y, labels = load_data(df_index_reference, df_files_to_load, TRAIN_DIR, 15)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"labels shape: {labels.shape}")
    print("=====================================")
    print(f"Number of unique speakers: {len(np.unique(labels))}")
    print(f"Number of unique classes: {len(np.unique(y))}")
    print(f"Unique classes: {np.unique(y)}")

if __name__ == '__main__':
    main()
