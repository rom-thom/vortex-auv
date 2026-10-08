import csv

import numpy as np


def load_data(file_path: str) -> dict[str, np.ndarray]:
    """
    Load a CSV with a header into a dictionary of 1D float arrays.

    input: file path, file is assumed to have names representing each column on top, separated by comma

    return: dictionary with its name as key and numpy arrays as values
    """
    with open(file_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        names = [name.strip() for name in next(reader, [])]
        if not names or any(not name for name in names):
            raise ValueError("CSV must have a nonempty name for every column.")
        if len(set(names)) != len(names):
            raise ValueError("CSV column names must be unique.")

        columns = {name: [] for name in names}
        for row in reader:
            if not row:
                continue
            if len(row) != len(names):
                raise ValueError(
                    f"CSV line {reader.line_num}: expected {len(names)} values, "
                    f"got {len(row)}."
                )
            for name, value in zip(names, row):
                try:
                    columns[name].append(float(value))
                except ValueError as error:
                    raise ValueError(
                        f"CSV line {reader.line_num}, column {name!r}: "
                        f"expected a number, got {value!r}."
                    ) from error

    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}













# don't mind me, just testin here
if __name__=="__main__":    
    import matplotlib.pyplot as plt

    data = load_data("system_identification/data/experiment.csv")
    time = data["time"]  # 1-D NumPy float array
    values = data["values"]

    plt.plot(time, values)
    plt.show()



