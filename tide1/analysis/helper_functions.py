from netCDF4 import Dataset
import numpy as np
from typing import List, Tuple, Union

# Dictionary mapping energy levels to their associated file chunks
ENERGY_LEVEL_FILES = {
    "wp50": ['wp50-t1.nc', 'wp50-t2.nc', 'wp50-t3.nc'],
    "wp60": ['wp60-t1.nc', 'wp60-t2.nc', 'wp60-t3.nc'],
    "wp75": ['wp75-t1.nc', 'wp75-t2.nc', 'wp75-t3.nc'],
    "wp80": ['wp80-t1.nc', 'wp80-t2.nc', 'wp80-t3.nc'],
    "wp90": ['wp90-t1.nc', 'wp90-t2.nc', 'wp90-t3.nc']
}

def get_train_test_files(test_level: str) -> Tuple[List[str], List[str]]:
    """
    Given a test energy level (e.g., 'wp75'), return lists of training and test filenames.
    """
    train_files = []
    for level, files in ENERGY_LEVEL_FILES.items():
        if level != test_level:
            train_files.extend(files)
    test_files = ENERGY_LEVEL_FILES[test_level]
    return train_files, test_files


def load_nc_files(data_dir: str, train_files: List[str], test_files: List[str]) -> Tuple[List[Dataset], List[Dataset]]:
    """
    Load train and test netCDF files.
    Returns opened netCDF4.Dataset objects as lists.
    """
    nctrains = [Dataset(f"{data_dir}{f}", 'r') for f in train_files]
    nctest = [Dataset(f"{data_dir}{f}", 'r') for f in test_files]
    return nctrains, nctest


def load_variable_from_nc(data_dir: str, filename: str, var_name: str,
                          rec_slice=slice(0, 150),
                          y_slice=slice(0, 720),
                          x_slice=slice(0, 256)) -> np.ndarray:
    """
    Load and slice a variable from a netCDF file.
    """
    with Dataset(f"{data_dir}{filename}", 'r') as nc:
        data = np.squeeze(nc.variables[var_name][:])
        return data[rec_slice, y_slice, x_slice]
