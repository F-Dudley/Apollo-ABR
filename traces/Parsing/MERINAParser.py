from pathlib import Path
import pandas as pd


from . import TraceFileParser, TraceParser, TraceSummary


@TraceParser(name="MERINAParser", accepted_sources=["merina"])
class MERINAParser(TraceFileParser):

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

        df = pd.read_csv(
            trace_file,
            sep=r"\s+",
            header=None,
            names=["timestamp_s", "throughput_mbps"],
            engine="python",
        )

        df.insert(0, "trace_id", trace_id)

        # Timestamps, normalised to sample range.
        df["timestamp_s"] = df["timestamp_s"] - df["timestamp_s"].min()

        # Convert Throughput to byte per second, currently in mbps
        df["throughput_bytes_per_s"] = df["throughput_mbps"] * 125000

        # Clean-up intermediate columns and sort by timestamp
        df.drop(columns=["throughput_mbps"], inplace=True)
        df.sort_values("timestamp_s", inplace=True)

        # Save Processed Trace to Output Directory
        self.save_file(df, output_file)

        return {
            "trace_id": trace_id,
            "split_type": split_type,
            "original_file": str(trace_file),
            "duration_s": df["timestamp_s"].max(),
            **self.get_additional_stats(
                df, bitrate_ladder_mbps, segment_duration_s, max_download_duration_s
            ),
        }
