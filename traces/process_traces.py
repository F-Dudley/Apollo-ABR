import os
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import hashlib
import argparse
import numpy as np

TRACES_ROOT = os.path.join(os.path.dirname(__file__))

TRACES_RAW_DIR = os.path.join(TRACES_ROOT, "raw")
TRACES_COOKED_DIR = os.path.join(TRACES_ROOT, "cooked")

from Parsing import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process ABR-Sim Trace Files")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=TRACES_RAW_DIR,
        help="Directory containing raw trace files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=TRACES_COOKED_DIR,
        help="Directory to save processed trace files and summary",
    )
    parser.add_argument(
        "--ladder-bitrates-mbps",
        type=float,
        nargs="+",
        default=[0.145, 1.600, 3.400, 5.800, 8.100, 16.800],
    )
    parser.add_argument("--segment-duration-s", type=int, default=5)
    parser.add_argument("--max-download-duration-s", type=int, nargs="+", default=4)
    parser.add_argument("--train-test-split", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=47)
    return parser.parse_args()


def hash_name(name: str, digest_size: int = 32) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(name.encode("utf-8"))

    return h.hexdigest()


def shuffle_traces(trace_files: list[Path], seed: int = 42) -> list[Path]:
    np.random.seed(seed)
    shuffled_files = trace_files.copy()
    np.random.shuffle(shuffled_files)
    np.random.seed(None)
    return shuffled_files


if __name__ == "__main__":

    args = parse_args()

    # Load All Trace files of format "timestamp (in seconds), throughput (in Megabits per second)"
    # Files are not with a header, per line seperated by a space.

    raw_dir = Path(TRACES_RAW_DIR)
    assert (
        raw_dir.is_dir()
    ), f"Raw trace directory '{raw_dir}' does not exist or is not a directory"

    output_dir = Path(TRACES_COOKED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_summaries = []

    for source_dir in raw_dir.glob("*"):
        if not source_dir.is_dir():
            continue

        # Find All Files in Source Direct, and Shuffle Ordering to ensure random distribution of sources in train/test sets.
        trace_files = [f for f in source_dir.glob("**/*") if f.is_file()]
        traces_files = shuffle_traces(trace_files, seed=args.random_seed)

        num_train_files = int(len(trace_files) * args.train_test_split)

        parser = ParserRegistry.get_parser(source_dir.name)
        if parser is None:
            raise ValueError(f"No parser registered for source '{source_dir.name}'")

        for i, trace_file in tqdm(
            enumerate(trace_files),
            desc=f"Processing '{source_dir.name}' Traces",
            total=len(trace_files),
        ):

            scenario_id = hash_name(str(trace_file.relative_to(raw_dir)))
            output_target = "train" if i < num_train_files else "val"

            file_output = output_dir / output_target
            file_output.mkdir(parents=True, exist_ok=True)

            try:
                trace_summary = parser.parse_file(
                    trace_id=scenario_id,
                    split_type=output_target,
                    trace_file=trace_file,
                    output_file=file_output / f"{scenario_id}.parquet",
                    bitrate_ladder_mbps=args.ladder_bitrates_mbps,
                    segment_duration_s=args.segment_duration_s,
                    max_download_duration_s=args.max_download_duration_s,
                )

                trace_summaries.append(trace_summary)
            except Exception as e:
                print(f"Error processing trace '{trace_file}'")
                raise e

    summary_df = pd.DataFrame(trace_summaries)
    summary_df.to_csv(output_dir / "trace_summary.csv", index=False)
