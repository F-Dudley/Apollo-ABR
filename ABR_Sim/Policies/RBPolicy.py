from typing import Callable
import numpy as np

from ABR_Sim.Core.Types import BitrateLadder, BitrateLadderEntry, SimulatorState, Action
from ABR_Sim.Policies import ABRPolicy, ABRPolicyClass


@ABRPolicyClass(name="RB")
class RateBasedPolicy(ABRPolicy):
    """Rate-based (RB) ABR Adaptation Policy based on the dash.js ThroughputRule.
    Throughput estimate: harmonic mean of the last `history_length` segment throughputs.
    Decision: the highest representation whose bitrate does not exceed
    the safety_factor * estimate (the lowest representation if none qualify).

    Buffer level and switching history are deliberately ignored, so the policy is purely
    throughput-driven and memoryless.
    """

    def __init__(self, scenario_config, seed: int | None = None):
        super().__init__(scenario_config, seed)

        self.history_length = 5  # harmonic-mean window (segments)
        self.safety_factor = 0.9  # 0.9 mirrors the dash.js ThroughputRule
        self.default_index = 0

        self.segment_duration_s = scenario_config.segment_duration_s

    def select_action(
        self,
        state_t: SimulatorState,
        ladder: BitrateLadder,
        *,
        segment_number: int,
        max_segment_number: int,
        segment_lookup: Callable[[int], BitrateLadder],
    ) -> Action:

        estimate_bytes_per_s = self._predict_throughput(
            np.array(list(state_t.last_throughputs_bytes_per_s), dtype=float)
        )
        if estimate_bytes_per_s is None:
            # No valid throughput history yet: start at the default (lowest) representation
            return self._get_action_from_entry(
                self.default_index, ladder.get_entry(self.default_index)
            )

        target_kbps = self.safety_factor * self._bytes_per_s_to_kbps(
            estimate_bytes_per_s
        )
        bitrates_kbps = np.array(
            [entry["bitrate_kbps"] for entry in ladder.entries], dtype=float
        )

        sustainable = np.flatnonzero(bitrates_kbps <= target_kbps)
        index = int(sustainable[-1]) if sustainable.size else self.default_index

        return self._get_action_from_entry(index, ladder.get_entry(index))

    def _predict_throughput(
        self, last_throughputs_bytes_per_s: np.ndarray
    ) -> float | None:
        valid = last_throughputs_bytes_per_s[
            np.isfinite(last_throughputs_bytes_per_s)
            & (last_throughputs_bytes_per_s > 0)
        ]
        if len(valid) == 0:
            return None

        valid = valid[-self.history_length :]
        return float(len(valid) / np.sum(1.0 / valid))

    def _kbps_to_bytes_per_s(self, kbps: float) -> float:
        return kbps * 1000 / 8

    def _bytes_per_s_to_kbps(self, bytes_per_s: float) -> float:
        return bytes_per_s * 8 / 1000

    def _get_action_from_entry(self, index: int, entry: BitrateLadderEntry) -> Action:
        return Action(
            bitrate_index=index,
            bitrate_kbps=entry["bitrate_kbps"],
            resolution_width=entry["resolution_width"],
            resolution_height=entry["resolution_height"],
            vmaf=entry["vmaf"],
            segment_size_bytes=entry["segment_size_bytes"],
        )
