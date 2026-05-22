import os
import argparse

from .Core.Interfaces import SegmentCatalog, ABRPolicy, TransitionInfoProvider
from .Core.Manifest import Manifest
from .Core.Types import ScenarioConfig


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


if __name__ == "__main__":

    args = parse_args()

    # Load the segment catalog
    segment_catalog = SegmentCatalog(args.segment_catalog)

    # Load the manifest
    manifest = Manifest(segment_catalog)

    # Create the scenario configuration
    config = ScenarioConfig(
        scenario_id="example_scenario",
        video_name=manifest.get_video_amount()[0],
        codec=manifest.get_codec_amount()[0],
        nic="WiFi",
        trace_id="example_trace",
    )

    # Initialize the ABR policy (replace with actual implementation)
    policy = ABRPolicy(seed=42)

    # Initialize transition info providers (replace with actual implementations)
    transition_info_providers: list[TransitionInfoProvider] = []

    # Create and run the simulator (replace with actual implementation)
    # simulator = ABRSimulator(config, segment_catalog, trace_provider, policy, buffer_manager, transition_info_providers)
    # simulator.run()
