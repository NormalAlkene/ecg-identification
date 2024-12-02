#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_mat(file_path: str) -> np.ndarray:
    """
    Load .mat file and return the data as numpy array.
    """
    return loadmat(file_path)["segmentData"]

def main():
    """
    1. Load: load the .mat file;
    2. Preprocess: detrend and denoise (cut off in the frequency domain);
    3. Preprocess: trim;
    4. Preprocess: normalize;
    5. Save: save the preprocessed data as .csv;
    6. Distribute: distribute the preprocessed data into train, validation and test sets;
    7. Save: save the distributed data as .pkl.
    """

if __name__ == "__main__":
    pass
