import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm
from collections import deque
from dataclasses import dataclass, asdict, is_dataclass

from ABR_Sim.Core.Interfaces import SegmentCatalog, ABRPolicy, TransitionInfoProvider
from ABR_Sim.Core.Manifest import Manifest, ManifestEntry
from ABR_Sim.Core.Types import ScenarioConfig, Transition
from ABR_Sim.Core.BufferManager import BufferManager

from ABR_Sim.Policies import PolicyRegistry
from ABR_Sim.Simulator import ABRSimulator
from ABR_Sim.SegmentCatalogs.StandardSegmentCatalog import StandardSegmentCatalog
from ABR_Sim.TraceProvider.StandardTraceProvider import StandardTraceProvider
from ABR_Sim.TransitionCollector import TransitionCollector

from ABR_Sim.TransitionProviders.SEEDEnergyInfoProvider import SEEDEnergyInfoProvider

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

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
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print detailed logs during simulation.",
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
    segment_catalog = StandardSegmentCatalog(segment_catalog_path)


def flatten_value(value: Any, prefix: str) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        flat = {}
        for sub_key, sub_value in value.items():
            if prefix:
                new_prefix = f"{prefix}_{sub_key}"
            else:
                new_prefix = str(sub_key)

            flat.update(flatten_value(sub_value, new_prefix))

        return flat

    elif isinstance(value, (list, tuple, deque, np.ndarray)):
        value = np.array(value)

        return {
            f"{prefix}_len": len(value),
            f"{prefix}_min": np.min(value) if value.size else None,
            f"{prefix}_max": np.max(value) if value.size else None,
            f"{prefix}_mean": np.mean(value) if value.size else None,
            f"{prefix}_delta": (value[-1] - value[0]) if value.size > 1 else None,
        }

    else:
        return {prefix: value}


def flatten_dataclass(instance) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise ValueError("Provided instance is not a dataclass.")

    flattened = {}

    # Flatten Nested Values (dataclasses, lists, dicts)
    for key, value in asdict(instance).items():
        flattened.update(flatten_value(value, key))

    return flattened


def flatten_transitions(transitions: list[Transition]) -> list[dict[str, Any]]:
    return [flatten_dataclass(transition) for transition in transitions]


def validate_transition_table(df: pd.DataFrame, config: ScenarioConfig) -> None:
    if df.empty:
        raise ValueError(f"Scenario {config.scenario_id} produced no transitions.")

    required = [
        "scenario_id",
        "step_t",
        "segment_number",
        "download_time_s",
        "buffer_s",
        "buffer_s_next",
        "throughput_bytes_per_s",
        "used_energy_ret",
    ]

    # missing = [col for col in required if col not in df.columns]
    missing = False

    if missing:
        raise ValueError(f"Missing transition columns: {missing}")

    if not df["step_t"].is_monotonic_increasing:
        raise ValueError("step_t is not monotonic increasing.")

    non_negative = [
        "download_time_s",
        "buffer_s",
        "buffer_s_next",
        "rebuffer_time_s",
        "wait_time_s",
        "throughput_bytes_per_s",
        "used_energy_ret",
        "used_energy_encstore",
        "used_energy_decoding",
        "used_energy_display",
    ]

    for col in non_negative:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")

            if values.isna().any():
                raise ValueError(f"{col} contains NaN values.")

            if not np.isfinite(values).all():
                raise ValueError(f"{col} contains non-finite values.")

            if (values < 0).any():
                raise ValueError(f"{col} contains negative values.")

    if "buffer_s_next" in df.columns:
        if df["buffer_s_next"].max() > config.max_buffer_s + 1e-6:
            raise ValueError("buffer_s_next exceeds max buffer.")


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

        # Flatten Transitions into Dicts

        transitions = flatten_transitions(transitions)

        transitions_table = pd.DataFrame(
            transitions,
        )

        validate_transition_table(transitions_table, scenario_config)

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
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }


def get_video_codec_uniques(segment_catalog_path: str) -> tuple[list[str], list[str]]:
    segment_catalog = StandardSegmentCatalog(segment_catalog_path)

    return (
        list(segment_catalog.get_uniques("video_name")),
        list(segment_catalog.get_uniques("codec")),
    )


def main():

    args = parse_args()
    print("ABR-Sim Configuration: ", args)

    os.makedirs(args.output_directory, exist_ok=True)

    videos, codecs = get_video_codec_uniques(args.segment_catalog)

    if args.verbose:
        print(f"Videos: {videos}")
        print(f"Codecs: {codecs}")
        print(f"Network Types: {args.network_types}")
        print(f"Policies: {PolicyRegistry.available_policies()}")

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

    # Scenarios
    if args.max_scenarios is not None:
        print(f"Limiting to first {args.max_scenarios} scenarios for testing.")
        manifest = list(manifest)[: args.max_scenarios]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = executor.map(
            lambda entry: validate_manifest_entry(entry, args.output_directory),
            manifest,
            chunksize=1000,
        )

        scenario_entries = [
            entry
            for entry in tqdm(results, total=len(manifest), desc="Validating scenarios")
            if entry is not None
        ]

    assert any(
        scenario_entries
    ), "Non-Valid Scenarios found. Please check the manifest and output directory for issues."

    if args.verbose:
        print(f"Total valid scenarios to run: {len(scenario_entries)}")

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


if __name__ == "__main__":
    main()
