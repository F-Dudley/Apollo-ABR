import os
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import hashlib

TRACES_ROOT = os.path.join(os.path.dirname(__file__))

TRACES_RAW_DIR = os.path.join(TRACES_ROOT, "raw")
TRACES_COOKED_DIR = os.path.join(TRACES_ROOT, "cooked")


def hash_name(name: str, digest_size: int = 32) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(name.encode("utf-8"))

    return h.hexdigest()


def load_raw_trace(trace_file: Path) -> pd.DataFrame:
    return pd.read_csv(
        trace_file,
        sep=r"\s+",
        header=None,
        names=["timestamp_s", "throughput_mbps"],
        engine="python",
    )


def process_traces(trace_dir: str, output_dir: str) -> None:

    trace_files = [f for f in Path(trace_dir).glob("**/*") if f.is_file()]

    print(f"Found {len(trace_files)} trace files in {trace_dir}")
    for trace_file in tqdm(trace_files, desc="Processing Trace Files"):
        relative_path = trace_file.relative_to(trace_dir)
        output_file_dir = Path(output_dir) / relative_path.parent
        output_file_dir.mkdir(parents=True, exist_ok=True)

        # Load Raw Trace
        df = load_raw_trace(trace_file)

        trace_id = hash_name(str(relative_path))

        df.insert(0, "trace_id", trace_id)

        # Normalize timestamps to start from 0
        df["timestamp_s"] = df["timestamp_s"] - df["timestamp_s"].min()

        # Convert throughput from Mbps to bytes per second
        df["throughput_bytes_per_s"] = df["throughput_mbps"] * 125000

        df.drop(columns=["throughput_mbps"], inplace=True)

        # Save Processed Trace
        df.to_parquet(output_file_dir / f"{trace_id}.parquet", index=False)


if __name__ == "__main__":

    # Load All Trace files of format "timestamp (in seconds), throughput (in Megabits per second)"
    # Files are not with a header, per line seperated by a space.

    # Not all Files have an extension, but if they do its .log.

    raw_dir = Path(TRACES_RAW_DIR)
    output_dir = Path(TRACES_COOKED_DIR)

    raw_subdirs = [d for d in raw_dir.glob("*") if d.is_dir()]
    for raw_subdir in raw_subdirs:
        relative_path = raw_subdir.relative_to(raw_dir)

        output_subdir = output_dir / relative_path
        output_subdir.mkdir(parents=True, exist_ok=True)

        process_traces(str(raw_subdir), str(output_subdir))
