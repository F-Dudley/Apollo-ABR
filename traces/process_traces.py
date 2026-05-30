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
    return parser.parse_args()


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


def required_throughput_mbps(
    bitrate_mbps: float, segment_duration_s: int, max_download_duration_s: int
) -> float:
    segment_size_megabits = bitrate_mbps * segment_duration_s
    required_throughput_mbps = segment_size_megabits / max_download_duration_s
    return required_throughput_mbps


def highest_supported_bitrate_mbps(
    throughput_mbps: float,
    bitrate_ladder_mbps: list[float],
    segment_duration_s: int,
    max_download_duration_s: int,
) -> float:
    for bitrate in sorted(bitrate_ladder_mbps, reverse=True):
        if throughput_mbps >= required_throughput_mbps(
            bitrate, segment_duration_s, max_download_duration_s
        ):
            return bitrate
    return 0.0


def add_sample_durations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp_s").reset_index(drop=True).copy()

    timestamps = df["timestamp_s"].to_numpy(dtype=float)

    if len(timestamps) <= 1:
        df["sample_duration_s"] = 1.0
        return df

    diffs = np.diff(timestamps)
    valid_diffs = diffs[diffs > 0]

    if len(valid_diffs) == 0:
        median_interval = 1.0
    else:
        median_interval = float(np.median(valid_diffs))

    durations = np.empty(len(timestamps), dtype=float)
    durations[:-1] = np.maximum(diffs, 1e-9)
    durations[-1] = median_interval

    df["sample_duration_s"] = durations

    return df


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return float("nan")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    threshold = q * cumulative[-1]

    return float(values[np.searchsorted(cumulative, threshold)])


def get_throughput_stats(
    df: pd.DataFrame,
    bitrate_ladder_mbps: list[float],
    segment_duration_s: int = 5,
    max_download_duration_s: int = 4,
) -> dict[str, float]:

    df = add_sample_durations(df)

    if "throughput_mbps" in df.columns:
        throughput_mbps = df["throughput_mbps"].to_numpy(dtype=float)
    else:
        throughput_mbps = (
            df["throughput_bytes_per_s"].to_numpy(dtype=float) * 8.0 / 1_000_000.0
        )

    durations = df["sample_duration_s"].to_numpy(dtype=float)

    mean_throughput_mbps = float(np.average(throughput_mbps, weights=durations))
    min_throughput_mbps = float(np.min(throughput_mbps))
    max_throughput_mbps = float(np.max(throughput_mbps))

    p05_throughput_mbps = weighted_quantile(throughput_mbps, durations, 0.05)
    p50_throughput_mbps = weighted_quantile(throughput_mbps, durations, 0.50)
    p95_throughput_mbps = weighted_quantile(throughput_mbps, durations, 0.95)

    stats = {
        "mean_throughput_mbps": mean_throughput_mbps,
        "min_throughput_mbps": min_throughput_mbps,
        "p05_throughput_mbps": p05_throughput_mbps,
        "p50_throughput_mbps": p50_throughput_mbps,
        "p95_throughput_mbps": p95_throughput_mbps,
        "max_throughput_mbps": max_throughput_mbps,
        "duration_s": float(np.sum(durations)),
        "num_samples": int(len(df)),
        "segment_duration_s": float(segment_duration_s),
        "max_download_duration_s": float(max_download_duration_s),
        "download_headroom_factor": float(segment_duration_s / max_download_duration_s),
    }

    for name, throughput in [
        ("mean", mean_throughput_mbps),
        ("min", min_throughput_mbps),
        ("p05", p05_throughput_mbps),
        ("p50", p50_throughput_mbps),
        ("p95", p95_throughput_mbps),
        ("max", max_throughput_mbps),
    ]:
        stats[f"{name}_supported_bitrate_mbps"] = highest_supported_bitrate_mbps(
            throughput,
            bitrate_ladder_mbps,
            segment_duration_s,
            max_download_duration_s,
        )

    for bitrate_mbps in bitrate_ladder_mbps:
        required = required_throughput_mbps(
            bitrate_mbps,
            segment_duration_s,
            max_download_duration_s,
        )

        supported = throughput_mbps >= required
        supported_duration_s = float(np.sum(durations[supported]))
        support_pct = 100.0 * supported_duration_s / max(float(np.sum(durations)), 1e-9)

        key = str(bitrate_mbps).replace(".", "_")
        stats[f"support_time_pct_{key}mbps"] = support_pct

    return stats


def process_traces(trace_dir: str, output_dir: str, args: argparse.Namespace) -> None:

    summary_data = []

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

        summary_data.append(
            {
                "trace_id": trace_id,
                "original_file": str(trace_file),
                "duration_s": df["timestamp_s"].max(),
                **get_throughput_stats(
                    df,
                    bitrate_ladder_mbps=args.ladder_bitrates_mbps,
                    segment_duration_s=args.segment_duration_s,
                    max_download_duration_s=args.max_download_duration_s,
                ),
            }
        )

        df.drop(columns=["throughput_mbps"], inplace=True)

        df.sort_values(by="timestamp_s", inplace=True)

        # Save Processed Trace
        df.to_parquet(output_file_dir / f"{trace_id}.parquet", index=False)

    # Save Summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "trace_summary.csv"), index=False)


if __name__ == "__main__":

    args = parse_args()

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

        process_traces(str(raw_subdir), str(output_subdir), args)
