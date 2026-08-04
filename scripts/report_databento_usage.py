import argparse
from pathlib import Path

import humanize


def get_file_size(path: Path) -> int:
    return path.stat().st_size

def human_readable_size(bytes_size: int) -> str:
    return humanize.naturalsize(bytes_size, binary=True)

def main():
    parser = argparse.ArgumentParser(description="Report Databento download usage.")
    parser.add_argument("--dir", type=str, default="data/raw/databento", help="Directory containing Databento parquet files.")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.is_dir():
        print(f"[ERROR] Directory {base_dir} does not exist.")
        return

    total_bytes = 0
    file_count = 0
    print("Databento download summary:\n")
    for file in sorted(base_dir.rglob("*.parquet")):
        size = get_file_size(file)
        total_bytes += size
        file_count += 1
        print(f"{file.name}\t{human_readable_size(size)}")

    print("\n----------------------------")
    print(f"Total files: {file_count}")
    print(f"Total size: {human_readable_size(total_bytes)} ({total_bytes} bytes)")

if __name__ == "__main__":
    main()
