import os
import pandas as pd
import numpy as np

from ..Core.Types import BitrateLadderEntry
from ..Core.Interfaces import TraceProvider

# Provider Expects CSV Files with the following columns:
# timestamp_s, throughput_bytes, signal_strength_dbm (optional), RSPI_dbm (optional), RSRP_dbm (optional)


class StandardTraceProvider(TraceProvider):

    def __init__(self, trace_file_path: str, allow_loop: bool = True):
        self.trace_file_path = trace_file_path
        self.allow_loop = allow_loop

        if not os.path.isfile(trace_file_path):
            raise FileNotFoundError(f"Trace file not found: {trace_file_path}")

        if trace_file_path.endswith(".csv"):

            self.trace_df = pd.read_csv(trace_file_path)
        elif trace_file_path.endswith(".parquet"):
            self.trace_df = pd.read_parquet(trace_file_path)
        else:
            raise ValueError(
                f"Unsupported trace file format: {trace_file_path}. Supported formats are .csv and .parquet"
            )

        self.trace_time_s = 0.0
        self.timestamps_s = self.trace_df["timestamp_s"].to_numpy(dtype=float)
        self.max_timestamp_s = np.max(self.timestamps_s)

    def download(
        self, start_time_s: float, wait_time_s: float, segment_size_bytes: float
    ) -> tuple[float, list[float]]:

        throughputs_bytes_per_s = []

        self.trace_time_s += wait_time_s
        download_start_trace_time_s = self.trace_time_s

        trace_idx_start = self._index_at_time(self.trace_time_s)

        remaining_bytes = segment_size_bytes

        # Experience Through-put Traces until the segment is fully downloaded
        while remaining_bytes > 1e-8:

            # Adjust Index to Provided Trace Times - traces might have different time steps, so we need to find the correct index for the current time
            self.current_idx = self._index_at_time(self.trace_time_s)
            row = self.trace_df.iloc[self.current_idx]

            throughput = row["throughput_bytes_per_s"]
            throughputs_bytes_per_s.append(throughput)

            time_needed_s = remaining_bytes / throughput

            used_time_s = min(
                1.0, time_needed_s
            )  # Use at most 1 second of the trace at a time

            remaining_bytes -= throughput * used_time_s
            self.trace_time_s += used_time_s  # Move forward in time by the used time

        download_time_s = self.trace_time_s - download_start_trace_time_s
        trace_idx_end = self._index_at_time(self.trace_time_s)

        trace_debug = {
            "trace_time_start_s": download_start_trace_time_s,
            "trace_time_end_s": self.trace_time_s,
            "trace_idx_start": trace_idx_start,
            "trace_idx_end": trace_idx_end,
            "trace_samples_used": len(throughputs_bytes_per_s),
        }

        return download_time_s, throughputs_bytes_per_s, trace_debug

    def _index_at_time(self, time_s: float) -> int:
        if self.allow_loop:
            time_s = time_s % self.max_timestamp_s

        idx = np.searchsorted(self.timestamps_s, time_s, side="right") - 1
        return min(max(idx, 0), len(self.timestamps_s) - 1)
