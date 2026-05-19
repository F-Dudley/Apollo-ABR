import os
import pandas as pd

from ..Core.Interfaces import TraceProvider

# Provider Expects CSV Files with the following columns:
# timestamp_s, throughput_bytes, signal_strength_dbm (optional), RSPI_dbm (optional), RSRP_dbm (optional)


class StandardTraceProvider(TraceProvider):

    def __init__(self, trace_file_path: str, allow_loop: bool = True):
        self.trace_file_path = trace_file_path
        self.allow_loop = allow_loop

        if not os.path.isfile(trace_file_path):
            raise FileNotFoundError(f"Trace file not found: {trace_file_path}")

        self.trace_df = pd.read_csv(trace_file_path)
        self.current_idx = 0

        self.has_signal_strength = "signal_strength_dbm" in self.trace_df.columns
        self.has_rsrp = "RSRP_dbm" in self.trace_df.columns
        self.has_rspi = "RSPI_dbm" in self.trace_df.columns

    def download(
        self, start_time_s: float, wait_time_s: float, segment_size_bytes: float
    ) -> tuple[float, list[float], list[float]]:

        download_time_s = 0.0
        throughputs_kbps = []
        signal_strength_dbm = []

        # Assuming 1 sample per second in the trace
        self.current_idx += int(wait_time_s)

        # Experience Through-put Traces until the segment is fully downloaded
        while segment_size_bytes > 1e-8:

            row = self.trace_df.iloc[self.current_idx]

            throughput = row["throughput_bytes"]
            throughputs_kbps.append(throughput / 1000.0)  # Convert to kbps

            if self.has_signal_strength:
                signal_strength = row["signal_strength_dbm"]
            elif self.has_rsrp:
                signal_strength = row["RSRP_dbm"]
            elif self.has_rspi:
                signal_strength = row["RSPI_dbm"]
            else:
                signal_strength = 0.0
                signal_strength_dbm.append(signal_strength)

            segment_size_bytes -= throughput
            download_time_s += 1.0

            self.current_idx += 1
            if self.current_idx >= len(self.trace_df) and not self.allow_loop:
                raise IndexError("End of trace reached and looping is disabled.")
            else:
                self.current_idx %= len(self.trace_df)

        return download_time_s, throughputs_kbps, signal_strength_dbm
