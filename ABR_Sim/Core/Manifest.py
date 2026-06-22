import os
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
    trace_files = {}

    for root_dir in Path(trace_directory).iterdir():
        if not root_dir.is_dir():
            continue

        split_name = root_dir.name
        trace_files[split_name] = []

        for trace_file in root_dir.glob("*.parquet"):
            if trace_file.is_file():
                trace_files[split_name].append(trace_file.resolve().as_posix())

    expected_splits = {"train", "val"}
    actual_splits = set(trace_files.keys())

    assert (
        expected_splits == actual_splits
    ), f"Expected trace directory to contain {expected_splits} subdirectories, but found: {actual_splits}. Please ensure the trace directory is structured correctly."

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
    network: str
    policy: str
    trace_file: str

    cfg_params: dict[str, Any] | None


class Manifest:

    def __init__(
        self,
        videos: list[str],
        codecs: list[str],
        networks: list[str] = ["Eth", "LTE"],
        policies: list[str] = ["Random", "RandomWalk", "Throughput", "BOLA", "WISH"],
        permutation_columns: list[str] = [
            "video_name",
            "target_split",
            "codec",
            "network",
            "policy",
            "trace_file",
            "cfg_params",
        ],
        cfg_params: dict[str, Any] | None = None,
        trace_directory: str = "./traces",
        fresh_manifest: bool = False,
    ):
        self.videos = videos
        self.codecs = codecs
        self.networks = networks
        self.policies = policies
        self.permutation_columns = permutation_columns
        self.trace_directory = trace_directory
        self.cfg_params: dict[str, Any] | None = cfg_params

        self._construct_manifest(fresh_manifest=fresh_manifest)
        self._validate_manifest()

    def _construct_manifest(self, fresh_manifest: bool = False):

        if not fresh_manifest and os.path.exists("simulation_manifest.csv"):
            print("Loading existing manifest from 'simulation_manifest.csv'...")
            self._manifest = pd.read_csv("simulation_manifest.csv")
            return

        permutations = []

        trace_files = collect_trace_files(self.trace_directory)

        cfg_permutations = (
            get_param_permutations(self.cfg_params) if self.cfg_params else [{}]
        )

        assert (
            trace_files is not None and len(trace_files) > 0
        ), f"No trace files found in directory '{self.trace_directory}'. Please ensure it contains valid .parquet trace files."

        for split, files in trace_files.items():

            permutations.extend(
                [
                    {
                        "scenario_id": generate_id(
                            split,
                            video,
                            codec,
                            network,
                            policy,
                            trace_file,
                            (
                                json.dumps(cfg_param, sort_keys=True)
                                if cfg_param
                                else None
                            ),
                        ),
                        "target_split": split,
                        "video_name": video,
                        "codec": codec,
                        "network": network,
                        "policy": policy,
                        "trace_file": trace_file,
                        "cfg_params": (
                            json.dumps(cfg_param, sort_keys=True) if cfg_param else None
                        ),
                    }
                    for video, codec, policy, network, trace_file, cfg_param in product(
                        self.videos,
                        self.codecs,
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
