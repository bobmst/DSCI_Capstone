
import pandas as pd
from pathlib import Path
import numpy as np
from util import load_pickle
from tqdm import tqdm

class DataPrep:
    def __init__(self, train_dir, index_path, num_classes=None):
        self.train_dir = train_dir
        self.index_path = index_path

        if not isinstance(self.train_dir, Path):
            self.train_dir = Path(train_dir)
        if not isinstance(self.index_path, Path):
            self.index_dir = Path(index_path)

        self.random_state = 2024
        self.num_classes = num_classes
        ##TODO: need to adjust to add in test data later

    def load(self):
        self.df_index_reference, self.df_files_to_load = self.prepare_index_data(self.index_path)
        self.X, self.y, self.labels = self.load_data(self.train_dir, verbose=False)

    def prepare_index_data(self, path):
        df_index = pd.read_pickle(path)

        df_index = df_index.merge(df_index.groupby('speaker').size().reset_index(name='n_samples_per_speaker'), on='speaker', how='left')

        df_index['pkl_filename'] = df_index['pkl_file_dir'].apply(lambda x: Path(x).name)
        df_index_unique = df_index[["file_index", 'speaker', 'pkl_filename', 'n_samples_per_speaker']].drop_duplicates()

        df_files_to_load = df_index_unique[["pkl_filename"]].drop_duplicates()
        df_files_to_load = df_files_to_load.reset_index().sort_values(by='index').reset_index(drop=True)

        df_index_reference = df_index_unique.merge(df_files_to_load, on='pkl_filename', how='left').sort_values(by=['index', 'file_index']).reset_index(drop=True)

        df_index_reference["speaker_full_array_index"] = np.array(range(len(df_index_reference)))
        df_index_reference["pkl_filename_index"] = df_index_reference["index"]
        df_index_reference = df_index_reference[['file_index', 'pkl_filename', 'pkl_filename_index',
                                                 'n_samples_per_speaker', 'speaker',
                                                 'speaker_full_array_index']]

        speaker_map = {speaker: i for i, speaker in enumerate(df_index_reference["speaker"].unique())}
        df_index_reference['speaker_class'] = df_index_reference['speaker'].map(speaker_map)

        if self.num_classes:
            speakers = df_index_reference[df_index_reference['n_samples_per_speaker'] == 256]['speaker'].sample(
                self.num_classes, random_state=self.random_state).values

            df_index_reference[f"speaker{self.num_classes}"] = df_index_reference['speaker'].apply(
                lambda x: 1 if x in speakers else 0)

            speaker_map = {speaker: i for i, speaker in enumerate(speakers)}
            df_index_reference[f"speaker{self.num_classes}_class"] = df_index_reference['speaker'].map(speaker_map)

        return df_index_reference, df_files_to_load

    def load_data(self, input_dir, verbose=False):
        X_ls = []
        filenames = []
        for index, row in tqdm(self.df_files_to_load.iterrows()):
            filename = row['pkl_filename']
            if verbose:
                print("Loading: ", filename)
            x = load_pickle(input_dir / filename)
            X_ls.append(x)
            filenames.append(filename)
        X = np.concatenate(X_ls, axis=0)

        labels = self.df_index_reference["speaker"].values
        y = self.df_index_reference["speaker_class"].values

        if self.num_classes:
            df_index_reference_sub = self.df_index_reference[self.df_index_reference[f"speaker{self.num_classes}"] ==
                                                             1].sort_values(by=["pkl_filename_index", "pkl_filename"]).copy()

            X = X[df_index_reference_sub["speaker_full_array_index"].values]

            labels = df_index_reference_sub["speaker"].values
            y = df_index_reference_sub[f"speaker{self.num_classes}_class"].values

            self.df_index_reference = df_index_reference_sub

        return X, y, labels

    def get_data_info(self):

        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"labels shape: {self.labels.shape}")
        print("=====================================")
        print(f"Number of unique speakers: {len(np.unique(self.labels))}")
        print(f"Number of unique classes: {len(np.unique(self.y))}")
        print(f"Unique classes: {np.unique(self.y)}")

    def get_data(self, labels=False):
        if labels:
            return self.X, self.y, self.labels
        else:
            return self.X, self.y


if __name__ == '__main__':
    # For testing purposes
    from util import DATA_ADDRESS

    TRAIN_DIR = Path(DATA_ADDRESS) / "preprocessed/train"
    INDEX_PATH = Path(DATA_ADDRESS) / "preprocessed/df_index_train.pkl"

    data = DataPrep(TRAIN_DIR, INDEX_PATH, 15)
    data.load()
    data.get_data_info()

    X, y, labels = data.get_data(labels=True)

    print("Loading data complete")








