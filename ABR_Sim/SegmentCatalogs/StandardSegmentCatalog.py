from ..Core.Interfaces import SegmentCatalog
from ..Core.Types import BitrateLadder, BitrateLadderEntry

import pandas as pd


class StandardSegmentCatalog(SegmentCatalog):

    def __init__(self, catalog_path: str):

        self.catalog_path = catalog_path
        self._catalog = pd.read_csv(catalog_path)

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

    def get_video_amount(self) -> list[str]:
        return self._catalog["video_name"].unique().tolist()

    def get_codec_amount(self) -> list[str]:
        return self._catalog["codec"].unique().tolist()

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
                vmaf=row["vmaf"],
                segment_size_bytes=row["encoded_segment_size_bytes"],
            )
            entries.append(entry)

        return BitrateLadder(segment_number=segment_number, entries=entries)
