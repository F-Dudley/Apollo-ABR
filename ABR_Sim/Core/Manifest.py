import os
import pandas as pd
from typing import TypedDict, Any, Generator
from itertools import product
from uuid import uuid4

from pathlib import Path

from hashlib import blake2b


def collect_trace_files(trace_directory: str, nic_type: str) -> list[str]:
    # Defined as Root -> NetworkType -> TraceFiles
    trace_files = []

    nic_directory = os.path.join(os.path.abspath(trace_directory), nic_type.lower())

    if not os.path.exists(nic_directory):
        raise FileNotFoundError(f"NIC directory '{nic_directory}' does not exist.")

    path_dir = Path(nic_directory)
    for trace_file in path_dir.glob("*.csv"):
        trace_files.append(trace_file.resolve().as_posix())


def generate_id(*args, digest_size: int = 64) -> str:
    hasher = blake2b(digest_size=digest_size)
    id_str = "::".join(str(arg) for arg in args)
    hasher.update(id_str.encode("utf-8"))
    return hasher.hexdigest()


class ManifestEntry(TypedDict):
    scenario_id: str
    video_name: str
    codec: str
    network: str
    policy: str
    trace_file: str


class Manifest:

    def __init__(
        self,
        videos: list[str],
        codecs: list[str],
        networks: list[str] = ["Eth", "LTE"],
        policies: list[str] = ["Random", "RandomWalk", "Throughput", "BOLA", "WISH"],
        permutation_columns: list[str] = ["video_name", "codec", "network", "policy"],
        trace_directory: str = "./traces",
        fresh_manifest: bool = False,
    ):
        self.videos = videos
        self.codecs = codecs
        self.networks = networks
        self.policies = policies
        self.permutation_columns = permutation_columns
        self.trace_directory = trace_directory

        self._construct_manifest(fresh_manifest=fresh_manifest)
        self._validate_manifest()

    def _construct_manifest(self, fresh_manifest: bool = False):

        if not fresh_manifest and os.path.exists("simulation_manifest.csv"):
            print("Loading existing manifest from 'simulation_manifest.csv'...")
            self._manifest = pd.read_csv("simulation_manifest.csv")
            return

        permutations = []

        for network in self.networks:

            trace_files = collect_trace_files(self.trace_directory, network)

            permutations.extend(
                [
                    {
                        "scenario_id": generate_id(
                            video, codec, network, policy, trace_file
                        ),
                        "video_name": video,
                        "codec": codec,
                        "network": network,
                        "policy": policy,
                        "trace_file": trace_file,
                    }
                    for video, codec, policy, trace_file in product(
                        self.videos, self.codecs, self.policies, trace_files
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
            yield row.to_dict()
