from typing import Any

from .Core.BufferManager import BufferManager

from .Core.Types import Action, SimulatorState, ScenarioConfig, Transition
from .Core.Interfaces import (
    ABRPolicy,
    SegmentCatalog,
    TraceProvider,
    TransitionInfoProvider,
)


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
            scenario_id=self.config.scenario_id,
            step_t=0,
            segment_number=0,
            sim_time_s=0.0,
            buffer_s=self.config.initial_buffer_s,
            buffer_kb=0,
            last_bitrate_index=self.config.initial_bitrate_index,
            throughput_mbps=self.config.initial_throughput_mbps,
            last_action=Action(bitrate=self.config.initial_bitrate_index, vmaf=0.0),
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

        action_t = self.policy.select_action(self.state, bitrate_ladder)

        _ladder_entry = bitrate_ladder.get_entry(action_t.bitrate_index)

        segment_info = self.catalog.lookup(
            self.config.video_name,
            self.config.codec,
            self.state.segment_number,
            _ladder_entry,
        )

        #
        # Download / Through-put Calculations
        encoded_segment_size_bits = segment_info["encoded_segment_size"] * 8

        pre_buffer_info = self.buffer_manager.prepare_download(
            buffer_s=self.state.buffer_s, sim_time_s=self.state.sim_time_s
        )

        download_time_s, throughput_kbps = self.trace_provider.download(
            start_time_s=self.pre_buffer_info.download_start_time_s,
            size_bits=encoded_segment_size_bits,
        )

        post_buffer_info = self.buffer_manager.complete_download(
            buffer_s_before_download=pre_buffer_info.buffer_s_before_download,
            download_start_time_s=pre_buffer_info.download_start_time_s,
            download_time_s=download_time_s,
            decoding_time_s=segment_info.get("decoding_time_s", 0.0),
        )

        next_segment_number = self.state.segment_number + 1
        done = next_segment_number >= self.total_segments

        outcome_t = {
            "scenario_id": self.state.scenario_id,
            "step_t": self.state.step_t,
            "segment_number": self.state.segment_number,
            # Network Info
            "download_time_s": download_time_s,
            "throughput_kbps": throughput_kbps,
            # Buffer Info
            "wait_time_s": pre_buffer_info.wait_time_s,
            "rebuffer_time_s": post_buffer_info.rebuffer_time_s,
            "buffer_s_next": post_buffer_info.buffer_s_next,
            # Timing Info
            "total_time_used_s": post_buffer_info.total_time_used_s,
        }

        if self.transition_info_providers:
            for provider in self.transition_info_providers:
                additional_info = provider.compute(
                    config=self.config,
                    state_t=self.state,
                    action_t=action_t,
                    segment=segment_info,
                    outcome_t=outcome_t,
                    state_t1=self._build_next_state(
                        current_state=self.state,
                        action_t=action_t,
                        outcome_t=outcome_t,
                        done=done,
                    ),
                )

                overlap = set(outcome_t).intersection(additional_info)
                if overlap:
                    raise KeyError(
                        f"Transition Info Provider '{provider.name}' returned keys that overlap with existing outcome_t keys: {overlap}"
                    )

                outcome_t.update(additional_info)

        transition = Transition(
            scenario_id=self.state.scenario_id,
            step_t=self.state.step_t,
            state_t=self.state,
            action_t=action_t,
            outcome_t=outcome_t,
            done=done,
        )

        self.state = self._build_next_state(
            current_state=self.state, action_t=action_t, outcome_t=outcome_t, done=done
        )

        return transition

    def _get_state_dict(self, state: SimulatorState | None = None) -> dict[str, Any]:
        if state is None:
            state = self.state

        if state is None:
            raise RuntimeError("Simulator not initialized. State is None.")

        return {
            "scenario_id": state.scenario_id,
            "step_t": state.step_t,
            "segment_number": state.segment_number,
            "sim_time_s": state.sim_time_s,
            "buffer_s": state.buffer_s,
            "buffer_fraction": state.buffer_s / self.config.max_buffer_s,
            "action": state.last_action,
            "throughput_mbps": state.last_throughput_mbps,
            "rebuffer_time_s": state.last_rebuffer_time_s,
            "done": state.done,
        }

    def _build_action_dict(
        self, action: Action, segment: dict[str, Any]
    ) -> dict[str, Any]:
        action_dict = {
            x: getattr(action, x) for x in action.__dataclass_fields__.keys()
        }

        # Add Segment Info to Action Dict
        action_dict.update(
            {
                "codec": self.config.codec,
                "encoded_segment_size": segment["encoded_segment_size"],
            }
        )

        return action_dict

    def _build_next_state(
        self,
        current_state: SimulatorState,
        action_t: Action,
        outcome_t: dict[str, Any],
        done: bool,
    ) -> SimulatorState:
        next_state = SimulatorState(
            scenario_id=current_state.scenario_id,
            step_t=current_state.step_t + 1,
            segment_number=current_state.segment_number + 1,
            sim_time_s=current_state.sim_time_s + outcome_t["total_time_used_s"],
            buffer_s=outcome_t["buffer_s_next"],
            done=done,
        )

        return next_state
