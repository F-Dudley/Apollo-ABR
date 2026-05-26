import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm
from dataclasses import dataclass

from .Core.Interfaces import SegmentCatalog, ABRPolicy, TransitionInfoProvider
from .Core.Manifest import Manifest, ManifestEntry
from .Core.Types import ScenarioConfig
from .Core.BufferManager import BufferManager

from .Policies import PolicyRegistry
from .Simulator import ABRSimulator
from .TraceProvider.StandardTraceProvider import StandardTraceProvider
from .TransitionCollector import TransitionCollector

from .TransitionProviders.SEEDEnergyInfoProvider import SEEDEnergyInfoProvider

from concurrent.futures import ProcessPoolExecutor, as_completed

segment_catalog: SegmentCatalog | None = None


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
    parser.add_argument(
        "--output-directory",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Path to the directory where simulation results will be stored.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() - 1 or 1,
        help="Maximum number of worker processes to use for parallel simulations.",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=5.0,
        help="Duration of each video segment in seconds.",
    )
    parser.add_argument(
        "--max-buffer",
        type=float,
        default=30.0,
        help="Maximum buffer size in seconds.",
    )
    parser.add_argument(
        "--initial-buffer",
        type=float,
        default=0.0,
        help="Initial buffer size in seconds.",
    )
    parser.add_argument(
        "--network-types",
        type=str,
        nargs="+",
        default=["Eth", "LTE"],
        help="List of network types to include in the simulations (e.g., Eth, LTE).",
    )
    parser.add_argument(
        "--fresh-manifest",
        action="store_true",
        help="Whether to generate a fresh manifest instead of loading an existing one.",
    )
    return parser.parse_args()


def validate_manifest_entry(entry: ManifestEntry, output_directory: str) -> bool:
    # Implement validation logic for the manifest entry
    # For example, check if the trace file exists, if the video name is valid, etc.
    if not os.path.exists(entry["trace_file"]):
        print(f"Trace file '{entry['trace_file']}' does not exist.")
        return False

    # Validate File of "scenario_id.*" does not already exist in the output directory to avoid overwriting results
    scenario_output_path = os.path.join(
        output_directory, f"{entry['scenario_id']}.parquet"
    )
    if os.path.isfile(scenario_output_path):
        print(
            f"Output file for scenario '{entry['scenario_id']}' already exists at '{scenario_output_path}'."
        )
        return False

    return True


@dataclass(frozen=True)
class SimConfig:
    segment_duration_s: float = 5.0
    max_buffer_s: float = 30.0
    initial_buffer_s: float = 0.0


def init_worker(segment_catalog_path: str):
    global segment_catalog
    segment_catalog = SegmentCatalog(segment_catalog_path)


def run_simulation(
    permutation: ManifestEntry,
    output_directory: str = "./results",
    sim_config: SimConfig = SimConfig(),
) -> dict[str, Any]:

    global segment_catalog

    scenario_config = ScenarioConfig(
        scenario_id=permutation["scenario_id"],
        video_name=permutation["video_name"],
        codec=permutation["codec"],
        nic=permutation["network"],
        trace_id=permutation["trace_file"],
        policy_name=permutation["policy"],
        segment_duration_s=sim_config.segment_duration_s,
        max_buffer_s=sim_config.max_buffer_s,
        initial_buffer_s=sim_config.initial_buffer_s,
    )

    trace_provider = StandardTraceProvider(permutation["trace_file"], allow_loop=True)

    policy = PolicyRegistry.create_policy(
        permutation["policy"], seed=scenario_config.scenario_id
    )

    # Create Buffer Manager
    buffer_manager = BufferManager(
        segment_duration_s=scenario_config.segment_duration_s,
        max_buffer_s=scenario_config.max_buffer_s,
        wait_for_space=True,
        include_decoding_time=False,
    )

    # Additional Info Gatherers
    info_providers: list[TransitionInfoProvider] = [SEEDEnergyInfoProvider()]

    sim = ABRSimulator(
        config=scenario_config,
        catalog=segment_catalog,
        trace_provider=trace_provider,
        policy=policy,
        buffer_manager=buffer_manager,
        transition_info_providers=info_providers,
    )

    transition_collector = TransitionCollector(
        simulator=sim,
    )

    try:
        transitions, transition_columns = transition_collector.run()

        # Save transitions to output directory
        os.makedirs(output_directory, exist_ok=True)

        output_path = os.path.join(
            output_directory, f"{scenario_config.scenario_id}.parquet"
        )

        transitions_table = pd.DataFrame(
            transitions,
            columns=transition_columns,
        )
        transitions_table.to_parquet(output_path, index=False)

        return {
            "scenario_id": scenario_config.scenario_id,
            "status": "success",
            "output_path": output_path,
            "num_transitions": len(transitions),
            "error": None,
        }

    except Exception as e:
        return {
            "scenario_id": scenario_config.scenario_id,
            "status": "failure",
            "output_path": None,
            "num_transitions": 0,
            "error": str(e),
        }


def get_video_codec_uniques(segment_catalog_path: str) -> tuple[list[str], list[str]]:
    segment_catalog = SegmentCatalog(segment_catalog_path)

    return (segment_catalog.get_video_list(), segment_catalog.get_codec_list())


if __name__ == "__main__":

    args = parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    videos, codecs = get_video_codec_uniques(args.segment_catalog)

    # Load the manifest
    manifest = Manifest(
        videos,
        codecs,
        networks=args.network_types,
        policies=PolicyRegistry.available_policies(),
        trace_directory=args.trace_directory,
        fresh_manifest=args.fresh_manifest,
    )

    sim_config = SimConfig(
        segment_duration_s=args.segment_duration,
        max_buffer_s=args.max_buffer,
        initial_buffer_s=args.initial_buffer,
    )

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(args.segment_catalog,),
    ) as executor:

        futures = [
            executor.submit(
                run_simulation, permutation, args.output_directory, sim_config
            )
            for permutation in manifest
            if validate_manifest_entry(permutation, args.output_directory)
        ]

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Simulations"
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Simulation failed with error: {e}")

    summary_path = os.path.join(args.output_directory, "run_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"Simulation run summary saved to '{summary_path}'.")
