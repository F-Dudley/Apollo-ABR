from typing import Any, TypedDict
from pathlib import Path

import pandas as pd
import numpy as np


class TraceSummary(TypedDict):
    trace_id: str
    split_type: str
    original_file: str
    duration_s: float


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
        throughput_mbps = df["throughput_bytes_per_s"].to_numpy(dtype=float) / 1e6

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


class TraceFileParser:

    def parse_file(
        self,
        trace_id: str,
        split_type: str,
        trace_file: Path,
        output_file: Path,
        bitrate_ladder_mbps: list[float] = [0.145, 1.600, 3.400, 5.800, 8.100, 16.800],
        segment_duration_s: int = 5,
        max_download_duration_s: int = 4,
    ) -> TraceSummary:
        """Parses a trace file and returns a summary of the trace along with"""

        raise NotImplementedError("Subclasses must implement this method")

    def get_additional_stats(
        self,
        trace_data: pd.DataFrame,
        bitrate_ladder_mbps: list[float] = [0.145, 1.600, 3.400, 5.800, 8.100, 16.800],
        segment_duration_s: int = 5,
        max_download_duration_s: int = 4,
    ) -> dict[str, Any]:
        df = add_sample_durations(trace_data)

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
            "download_headroom_factor": float(
                segment_duration_s / max_download_duration_s
            ),
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
            support_pct = (
                100.0 * supported_duration_s / max(float(np.sum(durations)), 1e-9)
            )

            key = str(bitrate_mbps).replace(".", "_")
            stats[f"support_time_pct_{key}mbps"] = support_pct

        return stats

    def save_file(self, trace_data: pd.DataFrame, output_file: Path) -> None:
        match output_file.suffix:
            case ".parquet":
                trace_data.to_parquet(output_file, index=False)
            case ".csv":
                trace_data.to_csv(output_file, index=False)
            case _:
                raise ValueError(
                    f"Unsupported output file format: {output_file.suffix}"
                )
