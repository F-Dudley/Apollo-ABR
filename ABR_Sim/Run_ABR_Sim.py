import os
import argparse
from pathlib import Path
from typing import Any

from .Core.Interfaces import SegmentCatalog, ABRPolicy, TransitionInfoProvider
from .Core.Manifest import Manifest, ManifestEntry
from .Core.Types import ScenarioConfig

from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ABR-Sim with the specified configuration."
    )
    parser.add_argument(
        "--segment-catalog",
        type=str,
        required=True,
        help="Path to the segment catalog CSV file.",
    )
    parser.add_argument(
        "--trace-directory",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "traces"),
        required=True,
        help="Path to the directory containing network trace files.",
    )
    return parser.parse_args()


def collect_trace_files(trace_directory: str, nic_type: str) -> list[str]:
    # Defined as Root -> NetworkType -> TraceFiles
    trace_files = []

    nic_directory = os.path.join(os.path.abspath(trace_directory), nic_type.lower())

    if not os.path.exists(nic_directory):
        raise FileNotFoundError(f"NIC directory '{nic_directory}' does not exist.")

    path_dir = Path(nic_directory)
    for trace_file in path_dir.glob("*.csv"):
        trace_files.append(trace_file.resolve().as_posix())


def run_simulation(permutation: ManifestEntry):

    global segment_catalog


if __name__ == "__main__":

    args = parse_args()

    _segment_catalog = SegmentCatalog(args.segment_catalog)

    videos = _segment_catalog.get_video_list()
    codecs = _segment_catalog.get_codec_list()

    # Load the manifest
    manifest = Manifest(videos, codecs, trace_directory=args.trace_directory)

    def init_worker():
        global segment_catalog
        segment_catalog = _segment_catalog

    with ProcessPoolExecutor(
        max_workers=os.cpu_count() - 1 or 1, initializer=init_worker
    ) as executor:

        future = [
            executor.submit(
                run_simulation,
                permutation,
            )
            for permutation in manifest
        ]
