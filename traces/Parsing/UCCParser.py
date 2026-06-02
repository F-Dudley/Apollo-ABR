from pathlib import Path
import pandas as pd

from . import TraceFileParser, TraceParser, TraceSummary


@TraceParser(name="UCCParser", accepted_sources=["ucc_4G", "ucc_5G"])
class UCCParser(TraceFileParser):

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

        df = pd.read_csv(trace_file)
        df.insert(0, "trace_id", trace_id)

        # Timestamps, Convert Type and Normalise to sample range.
        df["timestamp_s"] = df["Timestamp"].apply(self._extract_time_from_timestamp)
        df["timestamp_s"] = df["timestamp_s"] - df["timestamp_s"].min()

        # Convert Throughput to bytes per second, currently named "DL_bitrate" and in bits per second.
        df["throughput_bytes_per_s"] = df["DL_bitrate"] * 1000.0 / 8.0

        df = df[["trace_id", "timestamp_s", "throughput_bytes_per_s"]]

        df.sort_values("timestamp_s", inplace=True)

        self.save_file(df, output_file)

        return {
            "trace_id": trace_id,
            "original_file": str(trace_file),
            "duration_s": df["timestamp_s"].max(),
            "split_type": split_type,
            **self.get_additional_stats(
                df, bitrate_ladder_mbps, segment_duration_s, max_download_duration_s
            ),
        }

    def _extract_time_from_timestamp(self, timestamp_str: str) -> float:
        # Timestamp in Datetime format, e.g. "2017.11.30_16.48.26"

        dt = pd.to_datetime(timestamp_str, format="%Y.%m.%d_%H.%M.%S")
        return dt.timestamp()
