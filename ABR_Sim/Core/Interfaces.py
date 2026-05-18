from typing import Any, Protocol

from .Types import Action, SimulatorState, ScenarioConfig


class SegmentCatalog(Protocol):
    def num_segments(self, video_name: str, codec: str) -> int: ...

    def get_ladder(
        self, video_name: str, codec: str, segment_number: int
    ) -> list[dict[str, Any]]: ...

    def lookup(
        self, video_name: str, codec, segment_number: int, bitrate: int
    ) -> dict[str, Any]: ...


class TraceProvider(Protocol):

    def download(self, start_time_s: float, size_bits: float) -> tuple[float, float]:
        """
        Returns:
            download_time_s: float
            throughput_mbps: float
        """
        ...


class ABRPolicy(Protocol):
    def select_action(
        self, state_t: dict[str, Any], ladder: list[dict[str, Any]]
    ) -> Action: ...


class TransitionInfoProvider(Protocol):

    name: str

    def compute(
        self,
        *,
        config: ScenarioConfig,
        state_t: dict[str, Any],
        action_t: dict[str, Any],
        segment: dict[str, Any],
        outcome_t: dict[str, Any],
        state_t1: SimulatorState
    ) -> dict[str, Any]:
        """
        Computes transition information based on the provided parameters.

        Returns:
            dict[str, Any]: A dictionary containing the computed transition information. Too be merged into outcome_t.
        """
