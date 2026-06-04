import os
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm
from collections import deque
from functools import partial
from dataclasses import dataclass, asdict, is_dataclass

from ABR_Sim.Core.Interfaces import SegmentCatalog, ABRPolicy, TransitionInfoProvider
from ABR_Sim.Core.Manifest import Manifest, ManifestEntry
from ABR_Sim.Core.Types import SimConfig, ScenarioConfig, Transition
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
        "--chunk-size",
        type=int,
        default=1,
        help="Size of chunks to process in parallel.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print detailed logs during simulation.",
    )
    return parser.parse_args()


def validate_manifest_entry(
    entry: ManifestEntry, output_directory: str, verbose: bool = False
) -> tuple[bool, str]:
    # Implement validation logic for the manifest entry
    # For example, check if the trace file exists, if the video name is valid, etc.
    if not os.path.exists(entry["trace_file"]):
        if verbose:
            print(f"Trace file '{entry['trace_file']}' does not exist.")
        return False, ""

    # Validate File of "scenario_id.*" does not already exist in the output directory to avoid overwriting results
    scenario_output_path = os.path.join(
        output_directory, f"{entry['scenario_id']}.parquet"
    )
    if os.path.isfile(scenario_output_path):
        if verbose:
            print(
                f"Output file for scenario '{entry['scenario_id']}' already exists at '{scenario_output_path}'."
            )
        return False, entry["scenario_id"]

    return True, entry["scenario_id"]


def init_worker(segment_catalog_path: str):
    global segment_catalog
    segment_catalog = StandardSegmentCatalog(segment_catalog_path)


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.generic):
        a = a.item()

    if isinstance(b, np.generic):
        b = b.item()

    if a is None or b is None:
        return a is None and b is None

    try:
        if isinstance(a, float) and isinstance(b, float):
            if np.isnan(a) and np.isnan(b):
                return True

        return bool(a == b)

    except Exception:
        return False


def emit_leaf(
    *,
    flattened: dict[str, Any],
    seen: dict[str, list[dict[str, Any]]],
    raw_key: str,
    full_key: str,
    value: Any,
) -> None:
    """
    Emits a flattened leaf while avoiding duplicated same-key/same-value fields.

    Behaviour:
        first raw_key occurrence:
            emit as raw_key

        repeated raw_key with same value:
            skip

        repeated raw_key with different value:
            rename previous raw_key entry to its full_key
            emit current entry as full_key
    """
    previous_items = seen.get(raw_key, [])

    # If same raw key and same value already emitted, skip.
    for item in previous_items:
        if values_equal(item["value"], value):
            return

    # First time seeing this raw key: emit compact name.
    if not previous_items:
        flattened[raw_key] = value
        seen[raw_key] = [
            {
                "flat_key": raw_key,
                "full_key": full_key,
                "value": value,
            }
        ]
        return

    # Conflict: same raw key, different value.
    # Ensure previous compact entries are expanded.
    for item in previous_items:
        old_flat_key = item["flat_key"]
        previous_full_key = item["full_key"]

        if old_flat_key == raw_key:
            flattened.pop(old_flat_key, None)
            flattened[previous_full_key] = item["value"]
            item["flat_key"] = previous_full_key

    # Emit current conflicting value using full prefix.
    flattened[full_key] = value
    previous_items.append(
        {
            "flat_key": full_key,
            "full_key": full_key,
            "value": value,
        }
    )


def summarise_sequence(value: Any, prefix: str) -> dict[str, Any]:
    arr = np.asarray(value)

    if arr.size == 0:
        return {
            f"{prefix}_len": 0,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_delta": None,
        }

    if not np.issubdtype(arr.dtype, np.number):
        return {
            f"{prefix}_len": int(len(arr)),
        }

    return {
        f"{prefix}_len": int(len(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_delta": float(arr[-1] - arr[0]) if arr.size > 1 else None,
    }


def flatten_value(
    *,
    value: Any,
    prefix: str,
    raw_key: str,
    flattened: dict[str, Any],
    seen: dict[str, list[dict[str, Any]]],
) -> None:
    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            sub_key = str(sub_key)
            full_key = f"{prefix}_{sub_key}" if prefix else sub_key

            flatten_value(
                value=sub_value,
                prefix=full_key,
                raw_key=sub_key,
                flattened=flattened,
                seen=seen,
            )

        return

    if isinstance(value, (list, tuple, deque, np.ndarray)):
        summary = summarise_sequence(value, prefix)

        for summary_key, summary_value in summary.items():
            summary_suffix = summary_key.removeprefix(f"{prefix}_")

            emit_leaf(
                flattened=flattened,
                seen=seen,
                raw_key=f"{raw_key}_{summary_suffix}",
                full_key=summary_key,
                value=summary_value,
            )

        return

    emit_leaf(
        flattened=flattened,
        seen=seen,
        raw_key=raw_key,
        full_key=prefix,
        value=value,
    )


def flatten_dataclass(instance) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise ValueError("Provided instance is not a dataclass.")

    flattened: dict[str, Any] = {}
    seen: dict[str, list[dict[str, Any]]] = {}

    for key, value in asdict(instance).items():
        flatten_value(
            value=value,
            prefix=str(key),
            raw_key=str(key),
            flattened=flattened,
            seen=seen,
        )

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
        "decoding_time_s",
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
    start_time = time.perf_counter()

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
        permutation["policy"], sim_config=sim_config, seed=scenario_config.scenario_id
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
            "elapsed_time_s": time.perf_counter() - start_time,
            "num_transitions": len(transitions),
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

    except Exception as e:
        return {
            "scenario_id": scenario_config.scenario_id,
            "status": "failure",
            "output_path": None,
            "elapsed_time_s": time.perf_counter() - start_time,
            "num_transitions": 0,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc().replace("\n", " "),
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

    worker_fn = partial(
        validate_manifest_entry,
        output_directory=args.output_directory,
        verbose=args.verbose,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        scenario_entries = set()
        for requires_run, scenario_id in tqdm(
            executor.map(worker_fn, manifest, chunksize=args.chunk_size),
            total=len(manifest),
            desc="Validating Scenarios",
        ):
            if requires_run:
                scenario_entries.add(scenario_id)

    if len(scenario_entries) == 0:
        print("No valid scenarios to run. Exiting.")
        return

    if args.verbose:
        print(f"Total valid scenarios to run: {len(scenario_entries)}")

    filtered_manifest = [
        entry for entry in manifest if entry["scenario_id"] in scenario_entries
    ]

    worker_fn = partial(
        run_simulation,
        output_directory=args.output_directory,
        sim_config=sim_config,
    )

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(args.segment_catalog,),
    ) as executor:

        results = []
        for result in tqdm(
            executor.map(worker_fn, filtered_manifest, chunksize=args.chunk_size),
            total=len(filtered_manifest),
            desc="Simulations",
            smoothing=0.05,
            mininterval=1.0,
        ):
            try:
                results.append(result)
            except Exception as e:
                print(f"Simulation failed with error: {e}")

    pd.DataFrame(results).to_csv("run_summary.csv", index=False)
    print(f"Simulation run summary saved to '{os.path.abspath('run_summary.csv')}'.")


if __name__ == "__main__":
    main()
