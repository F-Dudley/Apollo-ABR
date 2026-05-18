from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str

    video_name: str
    codec: str
    trace_id: str
    policy_name: str

    segment_duration: float
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
    buffer_bytes: int

    last_bitrate_index: int
    throughput_mbps: float

    done: bool = False


@dataclass
class Action:
    bitrate: int
    vmaf: float


@dataclass
class Transition:
    scenario_id: str
    step_t: int

    state_t: dict[str, Any]
    action_t: dict[str, Any]
    outcome_t: dict[str, Any]

    state_t1: SimulatorState

    done: bool
