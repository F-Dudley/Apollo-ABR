from typing import Any

from .Core.BufferManager import BufferManager

from .Core.Types import (
    Action,
    SimulatorState,
    ScenarioConfig,
    Transition,
    BitrateLadder,
)
from .Core.Interfaces import (
    ABRPolicy,
    SegmentCatalog,
    TraceProvider,
    TransitionInfoProvider,
)

from collections import deque
import dataclasses
import numpy as np


class ABRSimulator:
    def __init__(
        self,
        config: ScenarioConfig,
        catalog: SegmentCatalog,
        trace_provider: TraceProvider,
        policy: ABRPolicy,
        buffer_manager: BufferManager,
        transition_info_providers: list[TransitionInfoProvider] | None = None,
    ):
        self.config = config
        self.catalog = catalog
        self.trace_provider = trace_provider
        self.policy = policy
        self.buffer_manager = buffer_manager
        self.transition_info_providers = transition_info_providers or []

        self.state: SimulatorState | None = None
        self.total_segments: int | None = None

    def reset(self) -> dict[str, Any]:
        self.total_segments = self.catalog.num_segments(
            self.config.video_name, self.config.codec
        )

        self.state = SimulatorState(
            config=self.config,
            scenario_id=self.config.scenario_id,
            step_t=0,
            segment_number=0,
            sim_time_s=0.0,
            buffer_s=self.config.initial_buffer_s,
            last_actions=deque(maxlen=5),
            last_throughputs_kbps=deque(maxlen=5),
            done=False,
        )

        return self._get_state_dict(self.state)

    def current_segment(self) -> int:
        if self.state is None:
            raise RuntimeError(
                "Initialize or Reset the simulator before calling current_segment()"
            )

        return self.state.segment_number

    def num_segments(self) -> int:
        if self.total_segments is None:
            raise RuntimeError(
                "Simulator not initialized. Please call init() before num_segments()."
            )

        return self.total_segments

    def step(self, action: Action | None = None) -> Transition:
        if self.state is None:
            raise RuntimeError(
                "Initialize or Reset the simulator before calling step()"
            )

        if self.state.done:
            raise RuntimeError(
                "Simulator has already reached a terminal state. Please reset before calling step() again."
            )

        if self.total_segments is None:
            raise RuntimeError(
                "Simulator not initialized. Please call init() before step()."
            )

        #
        # Current Segment / Action Selection
        bitrate_ladder = self.catalog.get_ladder(
            self.config.video_name, self.config.codec, self.state.segment_number
        )

        if action is not None:
            action_t = action
        else:
            action_t = self.policy.select_action(self.state, bitrate_ladder)

        _ladder_entry = bitrate_ladder.get_entry(action_t.bitrate_index)

        bitrate_ladder: BitrateLadder = self.catalog.get_ladder(
            self.config.video_name,
            self.config.codec,
            self.state.segment_number,
            _ladder_entry,
        )

        segment_info = bitrate_ladder.get_entry(action_t.bitrate_index)

        #
        # Download / Through-put Calculations
        encoded_segment_size_bytes = segment_info["encoded_segment_size_bytes"]

        pre_buffer_info = self.buffer_manager.prepare_download(
            buffer_s=self.state.buffer_s, sim_time_s=self.state.sim_time_s
        )

        download_time_s, throughput_traces_bytes_per_s = self.trace_provider.download(
            start_time_s=pre_buffer_info.download_start_time_s,
            wait_time_s=pre_buffer_info.wait_time_s,
            segment_size_bytes=encoded_segment_size_bytes,
        )

        post_buffer_info = self.buffer_manager.complete_download(
            buffer_s_before_download=pre_buffer_info.buffer_s_before_download,
            download_start_time_s=pre_buffer_info.download_start_time_s,
            download_time_s=download_time_s,
            decoding_time_s=segment_info.get("decoding_duration_s", 0.0),
        )

        info_t = {
            "scenario_id": self.state.scenario_id,
            "step_t": self.state.step_t,
            "segment_number": self.state.segment_number,
            # Network Info
            "download_time_s": download_time_s,
            "throughput_bytes_per_s": throughput_traces_bytes_per_s,
            # Buffer Info
            "wait_time_s": pre_buffer_info.wait_time_s,
            "rebuffer_time_s": post_buffer_info.rebuffer_time_s,
            "buffer_s_next": post_buffer_info.buffer_s_next,
            # Timing Info
            "total_time_used_s": post_buffer_info.total_time_used_s,
        }

        next_segment_number = self.state.segment_number + 1
        done = next_segment_number >= self.total_segments

        state_dict = self._get_state_dict(self.state)
        action_dict = self._build_action_dict(action_t)

        transition = Transition(
            scenario_id=self.state.scenario_id,
            step_t=self.state.step_t,
            state_t=state_dict,
            action_t=action_dict,
            info_t=info_t,
            done=done,
        )

        next_state = self._build_next_state(
            current_state=self.state, action_t=action_t, info_t=info_t, done=done
        )

        if self.transition_info_providers:
            for provider in self.transition_info_providers:
                additional_info = provider.compute(
                    config=self.config,
                    state_t=self.state,
                    action_t=action_t,
                    segment=segment_info,
                    info_t=info_t,
                    state_t1=next_state,
                )

                overlap = set(info_t).intersection(additional_info)
                if overlap:
                    raise KeyError(
                        f"Transition Info Provider '{provider.name}' returned keys that overlap with existing info_t keys: {overlap}"
                    )

                info_t.update(additional_info)

        self.state = next_state

        return transition

    def _get_state_dict(self, state: SimulatorState | None = None) -> dict[str, Any]:
        if state is None:
            state = self.state

        if state is None:
            raise RuntimeError("Simulator not initialized. State is None.")

        state_dict = dataclasses.asdict(state)

        del state_dict["last_actions"]
        del state_dict["last_throughputs_bytes_per_s"]

        return state_dict

    def _build_action_dict(self, action: Action) -> dict[str, Any]:
        return dataclasses.asdict(action)

    def _build_next_state(
        self,
        current_state: SimulatorState,
        action_t: Action,
        info_t: dict[str, Any],
        done: bool,
    ) -> SimulatorState:
        new_segment_number = current_state.segment_number + 1

        remaining_segments = (
            max(0, self.total_segments - new_segment_number)
            if self.total_segments is not None
            else 0
        )

        next_actions = current_state.last_actions.copy()
        next_actions.append(action_t)

        next_throughputs = current_state.last_throughputs_bytes_per_s.copy()
        next_throughputs.extend(info_t["throughput_bytes_per_s"])

        next_state = SimulatorState(
            scenario_id=current_state.scenario_id,
            step_t=current_state.step_t + 1,
            segment_number=new_segment_number,
            segments_remaining=remaining_segments,
            sim_time_s=current_state.sim_time_s + info_t["total_time_used_s"],
            buffer_s=info_t["buffer_s_next"],
            done=done,
            # -- Previous States
            last_actions=next_actions,
            last_throughputs_bytes_per_s=next_throughputs,
        )

        return next_state
