from typing import Any, Protocol

from .Types import (
    Action,
    SimulatorState,
    ScenarioConfig,
    BitrateLadder,
    BitrateLadderEntry,
)


class SegmentCatalog(Protocol):
    def __init__(self, catalog_path: str): ...

    def num_segments(self, video_name: str, codec: str) -> int: ...

    def get_ladder(
        self, video_name: str, codec: str, segment_number: int
    ) -> BitrateLadder: ...

    def get_video_list(self) -> list[Any]: ...

    def get_codec_list(self) -> list[Any]: ...


class TraceProvider(Protocol):

    def download(
        self,
        start_time_s: float,
        wait_time_s: float,
        segment_size_bytes: float,
    ) -> tuple[float, list[float], dict]:
        """
        Returns:
            download_time_s: float
            throughput_trace_kbps: list[float],
            signal_strength_dbm: list[float]
            debug_info: dict (can contain any additional information about the download, e.g. trace indices used, etc.)
        """
        ...


class ABRPolicy(Protocol):
    def __init__(self, scenario_config: ScenarioConfig, seed: int | None = None):
        self.seed = seed

    def select_action(
        self, state_t: SimulatorState, ladder: BitrateLadder
    ) -> Action: ...


class TransitionInfoProvider(Protocol):

    name: str

    def __init__(self, seed: int | None = None): ...

    def compute(
        self,
        config: ScenarioConfig,
        state_t: dict[str, Any],
        action_t: dict[str, Any],
        segment: dict[str, Any],
        info_t: dict[str, Any],
        state_t1: SimulatorState,
    ) -> dict[str, Any]:
        """
        Computes transition information based on the provided parameters.

        Returns:
            dict[str, Any]: A dictionary containing the computed transition information. Too be merged into info_t.
        """
        ...
