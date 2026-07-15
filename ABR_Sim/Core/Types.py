from __future__ import annotations

from dataclasses import asdict, dataclass, asdict
from typing import Any, Literal, Protocol, TypedDict, overload
from collections import deque

type NICType = Literal["Eth", "WiFi", "LTE", "5G"]


@dataclass(frozen=True)
class SimConfig:
    segment_duration_s: float = 5.0
    max_buffer_s: float = 30.0
    initial_buffer_s: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str

    video_name: str
    codec: str
    frame_rate: float
    nic: NICType
    trace_id: str
    policy_name: str

    segment_duration_s: float = 5.0
    max_buffer_s: float = 30.0
    initial_buffer_s: float = 0.0


@dataclass(frozen=True)
class SimulatorState:
    config: ScenarioConfig

    scenario_id: str

    step_t: int
    segment_number: int
    segments_remaining: int

    sim_time_s: float
    buffer_s: float

    last_actions: deque[Action]
    last_throughputs_bytes_per_s: deque[float]

    done: bool = False

    def __dict__(self):
        return asdict(self)


@dataclass(frozen=True)
class Action:
    bitrate_index: int
    bitrate_kbps: int
    resolution_width: str
    resolution_height: str
    vmaf: float

    segment_size_bytes: int


@dataclass
class Transition:
    scenario_id: str
    step_t: int

    state_t: dict[str, Any]
    action_t: dict[str, Any]

    info_t: dict[str, Any]

    done: bool = False


class BitrateLadderEntry(TypedDict):
    bitrate_kbps: int
    resolution_width: str
    resolution_height: str
    fps: int
    vmaf: float

    segment_size_bytes: int


@dataclass
class BitrateLadder:
    entries: list[BitrateLadderEntry]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> iter[BitrateLadderEntry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> BitrateLadderEntry:
        return self.entries[index]

    def get_entry(self, index: int) -> BitrateLadderEntry:
        if index < 0 or index >= len(self.entries):
            raise IndexError(
                f"Bitrate index {index} out of range for ladder of size {len(self.entries)}"
            )
        return self.entries[index]
