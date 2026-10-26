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











def merge_data(signals: dict[str, tuple[np.ndarray, np.ndarray]], target_time: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Merge data to same time and interpolates datapoints for those that has lower frequency
    
    signals: {data_name: (its_times, data_points), ...}
    target_time: the time we want to end with. if this is None we set it to the highest frequency 

    Assumes the same global time frame, and excludes time intervals not included in all the signals
    """
    if not signals:
        raise ValueError("At least one signal is required.")

    prepared = {}
    for name, (times, values) in signals.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip() or name == "time":
            raise ValueError("Signal names must be nonempty, trimmed strings other than 'time'.")
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        if times.ndim != 1 or values.ndim != 1:
            raise ValueError(f"Signal {name!r}: timestamps and values must be one-dimensional.")
        if times.size != values.size or times.size < 2:
            raise ValueError(f"Signal {name!r}: need matching lengths and at least two samples.")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError(f"Signal {name!r}: timestamps and values must be finite.")
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Signal {name!r}: timestamps must be strictly increasing.")
        prepared[name] = (times, values)

    start = max(times[0] for times, _ in prepared.values())
    end = min(times[-1] for times, _ in prepared.values())
    if start > end:
        raise ValueError("Signals have no overlapping time interval.")

    if target_time is None:
        target_time = min(
            (times for times, _ in prepared.values()),
            key=lambda times: np.median(np.diff(times)),
        )
    else:
        target_time = np.asarray(target_time, dtype=float)
        if target_time.ndim != 1 or not np.all(np.isfinite(target_time)):
            raise ValueError("target_time must be a finite one-dimensional array.")
        if np.any(np.diff(target_time) <= 0):
            raise ValueError("target_time must be strictly increasing.")

    time = target_time[(target_time >= start) & (target_time <= end)]
    if time.size == 0:
        raise ValueError("No target timestamps fall within the shared time interval.")

    merged = {"time": time}
    for name, (times, values) in prepared.items():
        merged[name] = np.interp(time, times, values)
    return merged







def merge_file_data(file_paths, exclude_comon_headers=False):
    """Merge CSV files, optionally dropping non-time columns shared by any files."""
    signals = {}
    excluded_headers = set()
    for file_path in file_paths:
        data = load_data(file_path)
        if "time" not in data:
            raise ValueError(f"CSV {file_path!r} must contain a 'time' column.")
        times = data.pop("time")
        for name, values in data.items():
            if name in excluded_headers:
                continue
            if name in signals:
                if exclude_comon_headers:
                    del signals[name]
                    excluded_headers.add(name)
                    continue
                raise ValueError(f"Duplicate signal name across CSV files: {name!r}.")
            signals[name] = (times, values)
    return merge_data(signals)





# don't mind me, just testin here
if __name__=="__main__":    

    signals = merge_file_data(
        ["system_identification/data/imu_test.csv", "system_identification/data/odom_test.csv", "system_identification/data/wrench_input_test.csv"],
        exclude_comon_headers=True,
    )

    print(signals.keys())





