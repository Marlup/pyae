import os
from glob import glob
from itertools import product
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex
from openpyxl import load_workbook, Workbook

from preprocessing.preprocessing import generate_noisy_signals, generate_synthetic_signal, min_max_scale, max_scale


class EMIDataReader():
    """
    EMIDataReader is a class designed to read and preprocess Electromechanical Impedance (EMI) data
    from Excel spreadsheets (.xlsx). It supports parsing sheets with structured sweep measurements,
    assembling multi-index DataFrames, and transforming the data into a format suitable for training
    machine learning models (including augmentation and normalization).

    Features:
    - Reads EMI data organized in Excel files with sweep sheets (e.g., "Sweep_1").
    - Converts data to pandas DataFrames with MultiIndex (load, sweep, sensor, freq_step).
    - Validates required structure (column names and index levels).
    - Converts EMI signals into structured NumPy arrays for modeling with optional augmentations.

    Attributes:
        data_path (str): Directory where EMI Excel files are stored.
        file_regex (str): Pattern for matching filenames (default: "*.xlsx").
        version_sep (str): Separator used to parse versioning from filenames.
        file_format (str): File extension (default: "xlsx").
    """

    def __init__(self, data_path: str, file_regex: str = "*.xlsx", version_sep: str = "_", file_format: str = "xlsx"):
        self.data_path = data_path
        self.file_regex = file_regex
        self.version_sep = version_sep
        self.file_format = file_format

    def read_all(self, stack_ranges: bool = True, n_features: int = 3) -> np.ndarray:
        """
        Reads all EMI Excel files in the directory and returns a stacked NumPy array.

        Parameters:
            stack_ranges (bool): Whether to stack all frequency ranges.
            n_features (int): Number of features per frequency point (e.g., freq, real, imag).

        Returns:
            np.ndarray: Array of shape depending on stack_ranges.
        """
        dataset = []
        file_list = sorted(glob(os.path.join(self.data_path, self.file_regex)),
                           key=lambda x: int(x.split(".")[0].split(self.version_sep)[-1]))
        for file_path in file_list:
            if not file_path.endswith(self.file_format):
                continue
            print(f"Reading: {file_path}")
            data = self._read_sheet(file_path, n_features=n_features, stack_ranges=stack_ranges)
            dataset.append(data)

        return np.array(dataset)

    def to_dataframe(
            self,
            stack_ranges: bool = True,
            column_names: Optional[List[str]] = None,
            index_names: Optional[List[str]] = None,
            n_features: int = 3
            ) -> DataFrame:
        """
        Reads EMI data and returns a pandas DataFrame with MultiIndex.

        Parameters:
            stack_ranges (bool): Whether to stack all frequency ranges.
            column_names (List[str]): Names of the data columns (default: None → ["frequency", "real", "imag"]).
            index_names (List[str]): Names for MultiIndex levels.
            n_features (int): Number of features per frequency point.

        Returns:
            pd.DataFrame: DataFrame of EMI signals with hierarchical index.
        """
        if column_names is None:
            column_names = ["frequency", "real", "imag"]

        data = self.read_all(stack_ranges=stack_ranges, n_features=n_features)

        if stack_ranges:
            n_loads, n_sweeps, n_sensors, n_steps, n_vars = data.shape
            index_names = index_names or ["load", "sweep", "sensor", "freq_step"]
            combs = product(range(n_loads), range(n_sweeps), range(n_sensors), range(n_steps))
        else:
            n_loads, n_sweeps, n_sensors, n_ranges, n_steps, n_vars = data.shape
            index_names = index_names or ["load", "sweep", "sensor", "freq_range", "freq_step"]
            combs = product(range(n_loads), range(n_sweeps), range(n_sensors), range(n_ranges), range(n_steps))

        return DataFrame(
            data.reshape(-1, n_vars),
            index=MultiIndex.from_tuples(list(combs), names=index_names),
            columns=column_names
        )
    
    def build_ndarray(self, df: pd.DataFrame, values: List[int], dim: int, **kwargs):
        """
        Processes EMI DataFrame signals into structured NumPy arrays with optional augmentations.

        Args:
            df (DataFrame): A validated EMI DataFrame.
            values (List[int]): Indices of the values to select from the signal features.
            dim (int): Axis along which to apply value selection.
            **kwargs: Parameters forwarded to `_build_ndarray_logic`, such as:
                - clip_to_positive (bool)
                - n_splits (int)
                - add_noise_augmentation (bool)
                - add_minmax_augmentation (bool)
                - normalization_mode (str)
                - on_load_target (bool)
                - on_ids (bool)

        Returns:
            np.ndarray or tuple: Augmented array and optionally targets/IDs.
        """
        required_levels = ["load", "sweep", "sensor"]
        for level in required_levels:
            if level not in df.index.names:
                raise ValueError(f"Missing required index level: '{level}'")

        expected_columns = {"frequency", "real", "imag"}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {expected_columns - set(df.columns)}")

        x = df.values.reshape(*[len(df.index.levels[i]) for i in range(df.index.nlevels)], -1)

        return self._build_ndarray_impl(x, values, dim, **kwargs)

    def _build_ndarray_impl(
        self,
        x,
        values,
        dim,
        clip_to_positive=True,
        n_splits=1,
        add_noise_augmentation=False,
        add_minmax_augmentation=False,
        probabilities_to_positive=None,
        normalization_mode="minmax",
        by_load_value=0,
        axis_min_max=-1,
        on_ids=False,
    ):
        axis_max = axis_min_max
        if n_splits < 1:
            n_splits = 1

        if clip_to_positive:
            x = x.clip(0.0, None)

        x = np.take(x, values, dim)
        *_, n_steps = x.shape

        augmentations = []

        if add_minmax_augmentation:
            print(f"\nRunning min-max augmentation, probs {probabilities_to_positive}:")
            print("\tShape of input data", x.shape)
            x_minmax_aug = generate_synthetic_signal(x, probabilities_to_positive)
            augmentations.append(x_minmax_aug)
            print("\tShape of output data:", x_minmax_aug.shape)

        if add_noise_augmentation:
            print("\nRunning noise augmentation:")
            print("\tShape of input data:", x.shape)
            x_noise_aug = generate_noisy_signals(x)
            print("\tShape of output data:", x_noise_aug.shape)
            augmentations.append(x_noise_aug)

        augmented_x = np.hstack([x, *augmentations])

        if normalization_mode == "minmax":
            augmented_x = min_max_scale(augmented_x, axis=axis_min_max, by_load_value=by_load_value)
        elif normalization_mode == "max":
            augmented_x = max_scale(augmented_x, axis=axis_max, by_load_value=by_load_value)

        new_n_steps = n_steps // n_splits
        *other_dims, _ = augmented_x.shape
        augmented_x = augmented_x.reshape(*other_dims, n_splits, new_n_steps)

        if normalization_mode == "minmax":
            augmented_x = min_max_scale(augmented_x, axis=axis_min_max)
        elif normalization_mode == "max":
            augmented_x = max_scale(augmented_x, axis=axis_max)

        if on_ids:
            ids = self.make_signal_ids(augmented_x)

        print("\nShape of the augmented data:", augmented_x.shape)

        if on_ids:
            return augmented_x, ids
        else:
            return augmented_x

    def _validate_dataframe(self, df: pd.DataFrame):
        """
        Ensures that required column names and index levels are present in the DataFrame.

        Raises:
            ValueError: If required structure is not found.
        """
        required_columns = {"frequency", "real", "imag"}
        required_indices = {"load", "sweep", "sensor"}

        if not required_columns.issubset(df.columns):
            raise ValueError(f"Expected columns {required_columns}, got {set(df.columns)}")

        if not required_indices.issubset(df.index.names):
            raise ValueError(f"Expected index levels {required_indices}, got {set(df.index.names)}")
    
    def make_signal_ids(self, data):
        """
        Builds a Tensor of categories which serves as id for each signal.

        Parameters:
            - data (ndarray -> (load, sweep, sensor, split, step))

        Returns:
            2D Tensor (ndarray) of categories. Each row.
        """
        
        n_loads, n_sweeps, n_sensors, n_splits, n_steps = data.shape
        load_vector = np.arange(n_loads)
        sweep_vector = np.arange(n_sweeps)
        sensor_vector = np.arange(n_sensors)
        split_vector = np.arange(n_splits)

        names = ["load", "sweep", "sensor", "split"]
        
        base_combinations = product(
            load_vector,
            sweep_vector,
            sensor_vector, 
            split_vector
            )

        combinations = list(base_combinations)
        
        return pd.MultiIndex.from_tuples(combinations, names=names)

    def _read_sheet(self, f_name: str, **kwargs) -> np.ndarray:
        workbook = load_workbook(f_name, read_only=True, data_only=True)
        sheets_data = []
        for sheet_name in workbook.sheetnames:
            if sheet_name[-1].isdigit():
                sheets_data.append(self._extract_sheet_data(workbook[sheet_name], **kwargs))
        return np.array(sheets_data)

    def _extract_sheet_data(self, ws: Workbook, **kwargs) -> np.ndarray:
        matrix_rows = []
        data = []
        for row in ws.iter_rows(values_only=True):
            if row and isinstance(row[0], (int, float)):
                matrix_rows.append(row)
                continue
            if matrix_rows:
                data.append(self._stack_features_and_ranges(matrix_rows, **kwargs))
                matrix_rows = []
        if matrix_rows:
            data.append(self._stack_features_and_ranges(matrix_rows, **kwargs))
        return np.array(data)

    def _stack_features_and_ranges(self, data, n_features: int = 3, stack_ranges: bool = True):
        data_array = np.array(data)
        stack = []
        for i in range(0, len(data_array[0]), n_features):
            stack.append(data_array[:, i:i+n_features])
        if stack_ranges:
            return np.vstack(stack)
        return np.array(stack)
