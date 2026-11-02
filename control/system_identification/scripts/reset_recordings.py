#!/usr/bin/env python3
"""Empty recording CSVs between experiments. Stop the ROS recorder first."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from system_identification.src.data_handeling import reset_ros_csv_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vehicle", choices=("sysid", "nautilus", "test_drone"), default="sysid",
        help="Dataset to clear (default: sysid).",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="Use this directory instead of the vehicle's default directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without clearing them.")
    args = parser.parse_args()

    directory = PROJECT_ROOT / "system_identification" / "data"
    if args.vehicle != "nautilus":
        directory /= args.vehicle
    if args.data_dir is not None:
        directory = args.data_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"Recording directory does not exist: {directory}")

    # Only recording files in this directory; leave other experiments untouched.
    paths = sorted(path for path in directory.glob("*_test.csv") if path.is_file())
    if args.dry_run:
        for path in paths:
            print(f"Would empty: {path}")
    else:
        try:
            for path in reset_ros_csv_files(paths):
                print(f"Emptied: {path}")
        except OSError as error:
            parser.exit(1, f"Could not reset all recordings: {error}\n")
    if not paths:
        print(f"No recording CSV files found in {directory}")


if __name__ == "__main__":
    main()
