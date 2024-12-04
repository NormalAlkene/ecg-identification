#!/usr/bin/env python3
"""
Preprocess
"""

import os
import shutil
import random
import multiprocessing

import numpy as np
import pandas as pd
import scipy

from utils.fileUtils import traverse_file
from configs import PATHS, PREPROCESSING


def load_mat(file_path: str) -> np.ndarray:
    """
    Load .mat file and return the data as numpy array.
    """
    return scipy.io.loadmat(file_path)["segmentData"].flatten()

class Detrender:
    """Detrender
    Detrend the data with np.polyfit.
    """
    def __init__(self, order: int = 1):
        self.order = order

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Detrend the data with np.polyfit.
        """
        coeffs = np.polyfit(np.arange(data.shape[0]), data, self.order)
        trend = np.polyval(coeffs, np.arange(data.shape[0]))
        return data - trend

class Denoiser:
    """Denoiser
    Denoise the data with FFT and IFFT
    """
    def __init__(self, fs, lowcut, highcut):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Cut off the data in the frequency domain.
        """
        freq = np.fft.fft(data)
        f = np.fft.fftfreq(data.shape[0], 1/self.fs)
        mask = (np.abs(f) < self.lowcut) | (np.abs(f) > self.highcut)
        freq[mask] = 0
        return np.fft.ifft(freq).real

class Normalizer:
    """Normalizer
    """
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize the data.
        """
        return (data - np.mean(data)) / np.std(data)

class Preprocessor:
    """Preprocessor
    """
    def __init__(self):
        self.preprocessors = [
            Detrender(order = PREPROCESSING.detrend_order),
            Denoiser(fs = PREPROCESSING.fs,
                     lowcut = PREPROCESSING.lowcut,
                     highcut = PREPROCESSING.highcut),
            Normalizer(),
        ]
    def __call__(self, data: np.ndarray) -> np.ndarray:
        for p in self.preprocessors:
            data = p(data)
        return data

def main():
    """
    1. Load: load the .mat file;
    2. Preprocess: detrend and denoise (cut off in the frequency domain);
    3. Preprocess: normalize;
    4. Save: save the preprocessed data as .csv;
    5. Distribute: distribute the preprocessed data into train, validation and test sets;
    6. Save: save the distributed data as .pkl.
    """

    # Preprocess
    preprocess = Preprocessor()

    file_paths = [ (d, f) for d, f in traverse_file(PATHS.path_ds_raw, ".mat") ]
    signals = [ load_mat(os.path.join(d, f)) for d, f in file_paths ]

    with multiprocessing.Pool() as pool:
        data = pool.map(preprocess, signals)

    # Group data by dir
    grouped: dict[str, list[np.ndarray]] = {}
    for idx, (directory, _) in enumerate(file_paths):
        cur_file = os.path.relpath(directory, PATHS.path_ds_raw)
        if cur_file not in grouped:
            grouped[cur_file] = []
        grouped[os.path.relpath(directory, PATHS.path_ds_raw)].append(data[idx])

    # Save data
    shutil.rmtree(PATHS.path_ds_preprocessed, ignore_errors=True)
    os.makedirs(PATHS.path_ds_preprocessed)
    for key, value in grouped.items():
        num_col = len(value)
        columns = ['t'] + [ str(i) for i in range(num_col)]

        t = np.arange(value[0].size) / PREPROCESSING.fs
        value.insert(0, t)

        df = pd.DataFrame(np.column_stack(value), columns = columns) # stacked by column
        df.to_csv(os.path.join(PATHS.path_ds_preprocessed, f"{key}.csv"), index = False)

    # Clear
    shutil.rmtree(PATHS.path_ds_training, ignore_errors=True)
    shutil.rmtree(PATHS.path_ds_validation, ignore_errors=True)
    shutil.rmtree(PATHS.path_ds_testing, ignore_errors=True)
    os.makedirs(PATHS.path_ds_training)
    os.makedirs(PATHS.path_ds_validation)
    os.makedirs(PATHS.path_ds_testing)

    # Distribute
    datasets = list(grouped.keys()) # already relpath
    len_total = len(datasets)
    cur_dataset = random.sample(datasets, int(len_total * PREPROCESSING.ratio_validation))
    for cur_file in cur_dataset:
        shutil.copy(os.path.join(PATHS.path_ds_preprocessed, f"{cur_file}.csv"),
                    os.path.join(PATHS.path_ds_validation, f"{cur_file}.csv"))

    datasets = list(set(datasets) - set(cur_dataset))
    cur_dataset = random.sample(datasets, int(len_total * PREPROCESSING.ratio_testing))
    for cur_file in cur_dataset:
        shutil.copy(os.path.join(PATHS.path_ds_preprocessed, f"{cur_file}.csv"),
                    os.path.join(PATHS.path_ds_testing, f"{cur_file}.csv"))

    datasets = list(set(datasets) - set(cur_dataset))
    cur_dataset = datasets
    for cur_file in cur_dataset:
        shutil.copy(os.path.join(PATHS.path_ds_preprocessed, f"{cur_file}.csv"),
                    os.path.join(PATHS.path_ds_training, f"{cur_file}.csv"))


if __name__ == "__main__":
    main()
