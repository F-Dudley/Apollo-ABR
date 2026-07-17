from ..Core.Interfaces import SegmentCatalog
from ..Core.Types import BitrateLadder, BitrateLadderEntry

import pandas as pd


class StandardSegmentCatalog(SegmentCatalog):

    def __init__(self, catalog_path: str, required_length: int | None = None):

        self.catalog_path = catalog_path

        self._catalog = self._load_catalog(catalog_path, required_length)

    def num_segments(self, video_name: str, codec: str) -> int:
        filtered = self._catalog[
            (self._catalog["video_name"] == video_name)
            & (self._catalog["codec"] == codec)
        ]
        if filtered.empty:
            raise ValueError(
                f"No segments found for video '{video_name}' with codec '{codec}' in catalog."
            )
        return len(filtered["segment_number"].unique())

    def get_uniques(self, column: str) -> list[str]:
        return self._catalog[column].unique()

    def get_ladder(
        self, video_name: str, codec: str, segment_number: int
    ) -> BitrateLadder:
        filtered = self._catalog[
            (self._catalog["video_name"] == video_name)
            & (self._catalog["codec"] == codec)
            & (self._catalog["segment_number"] == segment_number)
        ]
        if filtered.empty:
            raise ValueError(
                f"No ladder entries found for video '{video_name}', codec '{codec}', segment {segment_number} in catalog."
            )

        entries = []
        for _, row in filtered.iterrows():
            entry = BitrateLadderEntry(
                bitrate_kbps=row["bitrate_kbps"],
                resolution_width=row["resolution_width"],
                resolution_height=row["resolution_height"],
                fps=row["frame_rate"],
                vmaf=row["vmaf"],
                segment_size_bytes=row["encoded_segment_size_bytes"],
            )

            # Possible General Values
            entry.update(
                {
                    "encoding_duration_s": row.get("encoding_duration_s", 0.0),
                    "decoding_duration_s": row.get("decoding_duration_s", 0.0),
                }
            )

            # Energy Values (if available) - These are optional and may not be present in all catalogs, so we provide default values if they are missing.
            entry.update(
                {
                    "used_energy_encstore_j": row.get("used_energy_encstore_j", 0.0),
                    "used_energy_decoding_j": row.get("used_energy_decoding_j", 0.0),
                    "used_energy_display_j": row.get("used_energy_display_j", 0.0),
                }
            )
            entries.append(entry)

        return BitrateLadder(entries=entries)

    def _load_catalog(
        self, catalog_path: str, required_length: int | None = None
    ) -> pd.DataFrame:
        catalog = pd.read_csv(catalog_path)

        if required_length is not None:
            assert required_length > 0, "Required length must be a positive integer."

            # Filter Out Video Contents that do not have the required number of segments
            segment_counts = (
                catalog.groupby(["video_name", "codec"])["segment_number"]
                .nunique()
                .reset_index(name="segment_count")
            )

            valid_combinations = segment_counts[
                segment_counts["segment_count"] >= required_length
            ][["video_name", "codec"]]

            catalog = catalog.merge(valid_combinations, on=["video_name", "codec"])

        return catalog
