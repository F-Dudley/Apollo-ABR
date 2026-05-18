from types import Any

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

        self.initialized: bool = False

    def init(self):
        # Validate that the catalog has the required segments for the scenario
        self.total_segments = self.catalog.num_segments(
            self.config.video_name, self.config.codec
        )
        if self.total_segments == 0:
            raise ValueError(
                f"Segment catalog does not contain any segments for video {self.config.video_name} and codec {self.config.codec}"
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
        )

        self.initialized = True

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
        )

        return self._get_state_dict()

    def current_segment(self) -> int:
        if self.state is None:
            raise RuntimeError(
                "Initialize or Reset the simulator before calling current_segment()"
            )

        return self.state.segment_number

    def total_segments(self) -> int:
        if self.total_segments is None:
            raise RuntimeError(
                "Simulator not initialized. Please call init() before max_segments()."
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

        state_t = self._get_state_dict(self.state)

        ladder = self.catalog.get_ladder(
            self.config.video_name, self.config.codec, self.state.segment_number
        )

        # Get Action Dict
        if action is None:
            action = self.policy.select_action(state_t, ladder)

        action_t = {x: getattr(action, x) for x in action.__dataclass_fields__.keys()}

        # Calculate Segment Info
        segment = self.catalog.lookup(
            self.config.video_name,
            self.config.codec,
            self.state.segment_number,
            action.bitrate,
        )

        segment_size_bits = float(segment["encoded_segment_size"]) * 8.0

        # Wait for Buffer Space, if too full for the next segment request.
        prebuffer_info = self.buffer_manager.prepare_download(
            self.state.buffer_s, self.state.sim_time_s
        )

        # Download Segment after waiting for buffer space, if required.

        download_time_s, throughput_mbps = self.trace_provider.download(
            prebuffer_info.download_start_time_s, segment_size_bits
        )

        post_buffer_info = self.buffer_manager.complete_download(
            prebuffer_info.buffer_s_before_download,
            prebuffer_info.download_start_time_s,
            download_time_s,
        )

        next_throughput_estimate_mbps = self._estimate_throughput(
            self.state.throughput_mbps, throughput_mbps
        )

        next_segment_number = self.state.segment_number + 1
        done = next_segment_number >= self.total_segments

        next_state = SimulatorState(
            step_t=self.state.step_t + 1,
            segment_number=next_segment_number,
            sim_time_s=post_buffer_info.download_end_time_s,
            buffer_s=post_buffer_info.buffer_s_next,
            throughput_estimate_mbps=next_throughput_estimate_mbps,
            last_bitrate=action.bitrate,
            done=done,
        )

        state_t1 = self._get_state_dict(next_state)

        outcome_t = {
            "segment_number": self.state.segment_number,
            "encoded_segment_size": segment["encoded_segment_size"],
            # Network Info
            "download_time_s": download_time_s,
            "measured_throughput_mbps": throughput_mbps,
            # Buffer Info
            "buffer_s_before_wait": prebuffer_info.buffer_s_before_wait,
            "wait_time_s": prebuffer_info.wait_time_s,
            "buffer_s_before_download": prebuffer_info.buffer_s_before_download,
            "download_start_time_s": prebuffer_info.download_start_time_s,
            "rebuffer_time_s": post_buffer_info.rebuffer_time_s,
            "buffer_s_after_download": post_buffer_info.buffer_s_after_download,
            "buffer_s_next": post_buffer_info.buffer_s_next,
            "download_end_time_s": post_buffer_info.download_end_time_s,
        }

        if self.transition_info_providers:
            for provider in self.transition_info_providers:
                provider_data = provider.compute(
                    config=self.config,
                    state_t=state_t,
                    action_t=action_t,
                    segment=segment,
                    outcome_t=outcome_t,
                    state_t1=state_t1,
                )

                overlap = set(outcome_t).intersection(provider_data)
                if overlap:
                    raise KeyError(
                        f"TransitionInfoProvider {provider.name} produced keys that overlap with existing outcome_t keys: {overlap}"
                    )

                outcome_t.update(provider_data)

        # Generate The Final Transition
        transition = Transition(
            scenario_id=self.config.scenario_id,
            step_t=self.state.step_t,
            state_t=state_t,
            action_t=action_t,
            outcome_t=outcome_t,
            state_t1=state_t1,
            done=done,
        )

        self.state = next_state

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
            "throughput_estimate_mbps": state.throughput_estimate_mbps,
            "last_bitrate": state.last_bitrate,
        }

    def _estimate_throughput(
        self,
        previous_estimate_mbps: float,
        measured_throughput_mbps: float,
        alpha: float = 0.5,
    ) -> float:
        if previous_estimate_mbps is None or previous_estimate_mbps == 0.0:
            return measured_throughput_mbps

        return alpha * measured_throughput_mbps + (1 - alpha) * previous_estimate_mbps
