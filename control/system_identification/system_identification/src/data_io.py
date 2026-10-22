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





def write_to_file(data: dict[str, np.ndarray], file_path: str, append=False):
    """Write named columns to CSV

    append=False: replace the file with the supplied data.
    append=True: preserve existing rows and add new rows at the bottom.
    """
    names = [name.strip() for name in data]
    if not names or any(not name for name in names):
        raise ValueError("CSV must have a nonempty name for every column.")
    if len(set(names)) != len(names):
        raise ValueError("CSV column names must be unique.")

    columns = [np.asarray(values, dtype=float) for values in data.values()]
    if any(column.ndim != 1 for column in columns):
        raise ValueError("CSV columns must be one-dimensional.")
    if any(len(column) != len(columns[0]) for column in columns):
        raise ValueError("CSV columns must have the same length.")

    mode = "a+" if append else "w"
    with open(file_path, mode, newline="", encoding="utf-8") as file:
        needs_header = True
        needs_newline = False
        if append:
            file.seek(0)
            existing_names = next(csv.reader(file), None)
            if existing_names is not None:
                if existing_names != names:
                    raise ValueError("CSV columns must match the existing header and order.")
                needs_header = False
            file.seek(0, 2)
            size = file.tell()
            if size:
                file.buffer.seek(-1, 2)
                needs_newline = file.buffer.read(1) not in (b"\n", b"\r")
                file.seek(0, 2)
        writer = csv.writer(file)
        if needs_header:
            writer.writerow(names)
        elif needs_newline and len(columns[0]):
            file.write("\n")
        writer.writerows(zip(*columns))







# don't mind me, just testin here
if __name__=="__main__":    
    import matplotlib.pyplot as plt

    data = load_data("system_identification/data/experiment.csv")
    time = data["time"]  # 1-D NumPy float array
    values = data["velocity"]

    write_to_file(data, "system_identification/data/experiment_write.csv", append=False)

    plt.plot(time, values)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.show()


