from typing import Any, Protocol

from .Types import (
    Action,
    SimulatorState,
    ScenarioConfig,
    BitrateLadder,
    BitrateLadderEntry,
)


class SegmentCatalog(Protocol):
    def num_segments(self, video_name: str, codec: str) -> int: ...

    def get_ladder(
        self, video_name: str, codec: str, segment_number: int
    ) -> BitrateLadder: ...

    def lookup(
        self,
        video_name: str,
        codec: str,
        segment_number: int,
        ladder_entry: BitrateLadderEntry,
    ) -> BitrateLadder: ...


class TraceProvider(Protocol):

    def download(
        self, start_time_s: float, segment_size_bytes: float
    ) -> tuple[float, list[float], list[float]]:
        """
        Returns:
            download_time_s: float
            throughput_trace_kbps: list[float],
            signal_strength_dbm: list[float]
        """
        ...


class ABRPolicy(Protocol):
    def select_action(
        self, state_t: SimulatorState, ladder: BitrateLadder
    ) -> Action: ...


class TransitionInfoProvider(Protocol):

    name: str

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
