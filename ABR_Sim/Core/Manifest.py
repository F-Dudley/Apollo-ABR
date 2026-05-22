import pandas as pd
from itertools import product
from uuid import uuid4


class Manifest:

    def __init__(
        self,
        videos: list[str],
        codecs: list[str],
        networks: list[str] = ["Eth", "LTE"],
        policies: list[str] = ["Random", "RandomWalk", "Throughput", "BOLA", "WISH"],
    ):
        self.videos = videos
        self.codecs = codecs
        self.networks = networks
        self.policies = policies

        self._construct_manifest()
        self._validate_manifest()

    def _construct_manifest(self):
        permutations = [
            {
                "scenario_id": uuid4(),
                "video_name": video,
                "codec": codec,
                "network": network,
                "policy": policy,
            }
            for video, codec, network, policy in product(
                self.videos, self.codecs, self.networks, self.policies
            )
        ]

        # Convert the list of permutations into a DataFrame
        self._manifest = pd.DataFrame(permutations)

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

    def __iter__(self):
        for _, row in self._manifest.iterrows():
            scenario_config = {col: row[col] for col in self.permutation_columns}
            yield scenario_config,
