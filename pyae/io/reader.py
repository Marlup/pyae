import os
from glob import glob
from itertools import product
from typing import List, Optional

import numpy as np
from pandas import DataFrame, MultiIndex
from openpyxl import load_workbook, Workbook


class EMIDataReader():
    """
    A class to read Electromechanical Impedance (EMI) data stored in Excel spreadsheets.

    Provides methods to read EMI data into NumPy arrays or pandas DataFrames
    with MultiIndex including dimensions like load, sweep, sensor, and frequency step.

    Parameters:
        data_path (str): Directory path where EMI Excel files are located.
        file_regex (str): Regex pattern to match files (default: "*.xlsx").
        version_sep (str): Separator to split version from filename (default: "_").
        file_format (str): Extension of the Excel files to read (default: "xlsx").
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
