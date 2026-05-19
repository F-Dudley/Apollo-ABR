from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str

    video_name: str
    codec: str
    trace_id: str
    policy_name: str

    segment_duration_s: float
    max_buffer_s: float = 30.0
    initial_buffer_s: float = 0.0

    # Initial Parameters
    initial_bitrate_index: int = None
    initial_buffer_s: float = None
    initial_buffer_kb: int = None
    initial_throughput_mbps: float = None


@dataclass(frozen=True)
class SimulatorState:
    scenario_id: str

    step_t: int
    segment_number: int

    sim_time_s: float
    buffer_s: float

    done: bool = False


@dataclass(frozen=True)
class Action:
    bitrate_index: int


@dataclass
class Transition:
    scenario_id: str
    step_t: int

    state_t: SimulatorState
    action_t: Action

    outcome_t: dict[str, Any]

    done: bool


class BitrateLadderEntry(TypedDict):
    bitrate_kb: int
    resolution_width: str
    resolution_height: str
    vmaf: float


class BitrateLadder(TypedDict):
    segment_number: int
    entries: list[BitrateLadderEntry]

    def get_entry(self, bitrate_index: int) -> BitrateLadderEntry:
        if bitrate_index < 0 or bitrate_index >= len(self.entries):
            raise ValueError(
                f"Invalid bitrate index {bitrate_index} for segment {self.segment_number}. Valid range is [0, {len(self.entries) - 1}]."
            )

        return self.entries[bitrate_index]
