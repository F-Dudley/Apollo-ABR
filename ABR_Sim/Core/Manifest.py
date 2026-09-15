import os
from tokenize import group
import pandas as pd
from typing import TypedDict, Any, Generator
from itertools import product
from uuid import uuid4

from pathlib import Path

from hashlib import blake2b
import json


def collect_trace_files(
    trace_directory: str,
) -> dict[str, list[str]]:
    # Defined as Root -> Split -> TraceFiles

    trace_files: dict[str, list[str]] = {}
    trace_ext = {".parquet", ".csv"}

    for root_dir in Path(trace_directory).iterdir():
        if not root_dir.is_dir():
            continue

        trace_group = root_dir.name

        files = [
            trace_file.resolve().as_posix()
            for trace_file in root_dir.iterdir()
            if trace_file.is_file() and trace_file.suffix in trace_ext
        ]

        if files:
            trace_files[trace_group] = sorted(files)

    if not trace_files:
        raise ValueError(
            f"No trace files found in directory '{trace_directory}'. Please ensure it contains valid .parquet or .csv trace files organized in subdirectories."
        )

    return trace_files


def generate_id(*args, digest_size: int = 64) -> str:
    hasher = blake2b(digest_size=digest_size)
    id_str = "::".join(str(arg) for arg in args)
    hasher.update(id_str.encode("utf-8"))
    return hasher.hexdigest()


def get_param_permutations(sim_params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not sim_params:
        return [{}]

    keys = list(sim_params.keys())

    value_lists = [
        value if isinstance(value, list) else [value] for value in sim_params.values()
    ]

    expanded = []

    for values in product(*value_lists):
        expanded.append(dict(zip(keys, values)))

    return expanded


class ManifestEntry(TypedDict):
    scenario_id: str
    target_split: str
    video_name: str
    codec: str
    frame_rate: float
    network: str
    policy: str
    trace_file: str

    cfg_params: dict[str, Any] | None


class Manifest:

    def __init__(
        self,
        videos: list[str],
        codecs: list[str],
        frame_rates: list[float],
        networks: list[str] = ["Eth", "LTE"],
        policies: list[str] = ["Random", "RandomWalk", "Throughput", "BOLA", "WISH"],
        permutation_columns: list[str] = [
            "video_name",
            "target_split",
            "network",
            "policy",
            "trace_file",
            "cfg_params",
        ],
        content_assignments: dict[str, list[str]] | None = None,
        cfg_params: dict[str, Any] | None = None,
        trace_directory: str = "./traces",
        fresh_manifest: bool = False,
        verbose: bool = False,
    ):
        self.videos = videos
        self.codecs = codecs
        self.frame_rates = frame_rates
        self.networks = networks
        self.policies = policies
        self.permutation_columns = permutation_columns
        self.trace_directory = trace_directory
        self.content_assignments = content_assignments
        self.cfg_params: dict[str, Any] | None = cfg_params

        # Video Splitting into Train and Validation Sets
        # self._video_map = self._split_videos(train_videos, val_videos)

        self._construct_manifest(fresh_manifest=fresh_manifest, verbose=verbose)
        self._validate_manifest()

    def _construct_manifest(self, fresh_manifest: bool = False, verbose: bool = False):

        if not fresh_manifest and os.path.exists("simulation_manifest.csv"):
            print("Loading existing manifest from 'simulation_manifest.csv'...")
            self._manifest = pd.read_csv("simulation_manifest.csv")
            return

        permutations = []

        trace_files = collect_trace_files(self.trace_directory)
        video_map = self._get_video_map(list(trace_files.keys()))

        if verbose:
            print(f"Trace files found in '{self.trace_directory}':")
            for split, files in trace_files.items():
                print(f"- {split}: {len(files)} files")

        cfg_permutations = (
            get_param_permutations(self.cfg_params) if self.cfg_params else [{}]
        )

        if verbose:
            print(f"Configuration parameter permutations: {len(cfg_permutations)}")
            for idx, cfg_param in enumerate(cfg_permutations):
                print(f"  Permutation {idx + 1}: {cfg_param}")

        assert (
            trace_files is not None and len(trace_files) > 0
        ), f"No trace files found in directory '{self.trace_directory}'. Please ensure it contains valid .parquet trace files."

        for trace_group, files in trace_files.items():
            group_videos = video_map.get(trace_group, None)
            assert (
                group_videos is not None
            ), f"No videos found for trace group '{trace_group}'."

            permutations.extend(
                [
                    {
                        "scenario_id": generate_id(
                            trace_group,
                            video,
                            codec,
                            frame_rate,
                            network,
                            policy,
                            trace_file,
                            json.dumps(cfg_param, sort_keys=True),
                        ),
                        "target_group": trace_group,
                        "video_name": video,
                        "codec": codec,
                        "frame_rate": frame_rate,
                        "network": network,
                        "policy": policy,
                        "trace_file": trace_file,
                        "cfg_params": (
                            json.dumps(cfg_param, sort_keys=True) if cfg_param else None
                        ),
                    }
                    for video, codec, frame_rate, policy, network, trace_file, cfg_param in product(
                        group_videos,
                        self.codecs,
                        self.frame_rates,
                        self.policies,
                        self.networks,
                        files,
                        cfg_permutations,
                    )
                ]
            )

        self._manifest = pd.DataFrame(permutations)
        self._manifest.to_csv("simulation_manifest.csv", index=False)

    def _validate_manifest(self):

        for col in self.permutation_columns:
            if col not in self._manifest.columns:
                raise ValueError(f"Permutation column '{col}' not found in manifest.")

            if self._manifest[col].isnull().any():
                raise ValueError(
                    f"Permutation column '{col}' contains null values in manifest."
                )

        permutation_amount = len(
            self._manifest[self.permutation_columns].drop_duplicates()
        )
        if permutation_amount != len(self._manifest):
            raise ValueError(
                f"Manifest contains duplicate entries for the specified permutation columns. Found {permutation_amount} unique permutations but {len(self._manifest)} total entries."
            )

        print(
            f"Manifest loaded successfully with {len(self._manifest)} entries and {permutation_amount} unique permutations based on columns: {', '.join(self.permutation_columns)}."
        )

    def __iter__(self) -> Generator[ManifestEntry, None, None]:
        for _, row in self._manifest.iterrows():
            row = row.to_dict()
            row["cfg_params"] = (
                json.loads(row["cfg_params"]) if row["cfg_params"] else None
            )

            yield row

    def __len__(self) -> int:
        return len(self._manifest)

    def _get_video_map(self, trace_groups: list[str]) -> dict[str, list[str]]:

        video_set = set(self.videos)
        trace_group_set = set(trace_groups)

        unknown_groups = set(self.content_assignments.keys()) - trace_group_set
        if unknown_groups:
            raise ValueError(
                f"Content assignments contain unknown trace groups: {unknown_groups}. Available trace groups: {trace_group_set}."
            )

        video_map: dict[str, list[str]] = {}

        for trace_group in trace_groups:
            assigned_videos = self.content_assignments.get(trace_group)
            if assigned_videos is None:
                video_map[trace_group] = list(self.videos)
                continue

            unknown_videos = set(assigned_videos) - video_set
            if unknown_videos:
                raise ValueError(
                    f"Content assignments for trace group '{trace_group}' contain unknown videos: {unknown_videos}. Available videos: {video_set}."
                )

            video_map[trace_group] = assigned_videos

        return video_map
