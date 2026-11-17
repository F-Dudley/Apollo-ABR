from typing import Callable
from collections import deque
import numpy as np
from itertools import product

from ABR_Sim.Core.Types import BitrateLadder, BitrateLadderEntry, SimulatorState, Action
from ABR_Sim.Policies import ABRPolicy, ABRPolicyClass


@ABRPolicyClass(name="RobustMPC")
class RobustMPCPolicy(ABRPolicy):

    def __init__(self, scenario_config, seed: int | None = None):
        super().__init__(scenario_config, seed)

        self.lookahead = 5
        self.history_length = 5
        self.default_index = 0

        self._prev_prediction_bytes_per_s: float = None
        self._pred_errors: deque[float] = deque(maxlen=self.history_length)

        # QoE Parameters
        self.rebuffer_penalty = 4.3
        self.switch_penalty = 1.0

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

        lookahead = min(self.lookahead, max_segment_number - segment_number)

        if len(state_t.last_actions) == 0 or lookahead == 0:
            # If No History, Select Default Bitrate
            return self._get_action_from_entry(
                self.default_index, ladder.get_entry(self.default_index)
            )

        # Get numpy arrays of bitrates and sizes for the current ladder, including lookahead segments
        bitrates = (
            np.array([entry["bitrate_kbps"] for entry in ladder.entries]) / 1000.0
        )  # Mbps
        sizes = self._get_lookahead_sizes(segment_number, lookahead, segment_lookup)

        num_actions = len(bitrates)
        if sizes.shape[1] != num_actions:
            raise ValueError(
                f"Sizes shape {sizes.shape} does not match number of actions {num_actions}"
            )

        robust_throughput_bytes_per_s = self._predict_robust_throughput(
            np.array(list(state_t.last_throughputs_bytes_per_s))
        )
        if robust_throughput_bytes_per_s is None:
            return self._get_action_from_entry(
                self.default_index, ladder.get_entry(self.default_index)
            )

        previous_bitrate_index = (
            state_t.last_actions[-1].bitrate_index
            if len(state_t.last_actions) > 0
            else self.default_index
        )

        best_qoe = -np.inf
        best_action_index = self.default_index

        for action_sequence in product(range(num_actions), repeat=lookahead):
            qoe = self._calculate_qoe(
                action_sequence,
                state_t,
                bitrates,
                sizes,
                robust_throughput_bytes_per_s,
                previous_bitrate_index,
            )

            if qoe > best_qoe:
                best_qoe = qoe
                best_action_index = action_sequence[0]

        return self._get_action_from_entry(
            best_action_index, ladder.get_entry(best_action_index)
        )

    def _calculate_qoe(
        self,
        sequence,
        state_t,
        bitrates_kbps: np.ndarray,
        future_segment_sizes: np.ndarray,
        predicted_throughput_bytes_per_s: float,
        last_bitrate_index: int,
    ) -> float:

        qoe = 0.0
        ahead_buffer_s = state_t.buffer_s
        prev_bitrate_index = last_bitrate_index

        for i, bitrate_idx in enumerate(sequence):
            bitrate_kbps = bitrates_kbps[bitrate_idx]
            prev_bitrate_kbps = bitrates_kbps[prev_bitrate_index]

            segment_size_bytes = future_segment_sizes[i, bitrate_idx]

            # Avoid division by zero or negative throughput predictions
            predicted_throughput_bytes_per_s = max(
                predicted_throughput_bytes_per_s, 1e-9
            )

            download_time_s = segment_size_bytes / predicted_throughput_bytes_per_s

            rebuffer_time_s = max(0.0, download_time_s - ahead_buffer_s)

            ahead_buffer_s = (
                max(0.0, ahead_buffer_s - download_time_s) + self.segment_duration_s
            )

            # Accumulate QoE, using Bitrate as Utility.
            smoothness_penalty = self.switch_penalty * abs(
                bitrate_kbps - prev_bitrate_kbps
            )

            rebuffer_penalty = self.rebuffer_penalty * rebuffer_time_s

            qoe += bitrate_kbps - smoothness_penalty - rebuffer_penalty
            prev_bitrate_index = bitrate_idx

        return qoe

    def _predict_robust_throughput(
        self,
        last_throughputs_bytes_per_s: np.ndarray,
    ) -> float | None:

        valid = last_throughputs_bytes_per_s[
            np.isfinite(last_throughputs_bytes_per_s)
            & (last_throughputs_bytes_per_s > 0)
        ]

        if len(valid) == 0:
            return None

        # Limit to the most recent N samples.
        valid = valid[-self.history_length :]

        # Update prediction error using the newest real measurement.
        if self._prev_prediction_bytes_per_s is not None:
            latest = valid[-1]

            error = abs(self._prev_prediction_bytes_per_s - latest) / latest

            self._pred_errors.append(float(error))

        # Harmonic mean throughput estimate.
        harmonic_throughput = len(valid) / np.sum(1.0 / valid)

        # Robust correction using worst recent prediction error.
        max_error = max(self._pred_errors) if len(self._pred_errors) > 0 else 0.0

        robust_throughput = harmonic_throughput / (1.0 + max_error)

        # Pensieve stores the raw harmonic estimate for
        # comparison against the next observed throughput.
        self._prev_prediction_bytes_per_s = harmonic_throughput

        return float(robust_throughput)

    def _get_lookahead_sizes(
        self,
        segment_number: int,
        lookahead: int,
        segment_lookup: Callable[[int], BitrateLadder],
    ) -> np.ndarray:

        ref_ladder = segment_lookup(segment_number)
        ref_keys = [
            e["bitrate_kbps"] for e in ref_ladder.entries
        ]  # or whatever uniquely IDs a rung

        sizes = np.full((lookahead, len(ref_keys)), np.nan)
        for i in range(lookahead):
            ladder = segment_lookup(segment_number + i)
            by_key = {
                e["bitrate_kbps"]: e["segment_size_bytes"] for e in ladder.entries
            }
            for j, k in enumerate(ref_keys):
                if k in by_key:
                    sizes[i, j] = by_key[k]

        if np.isnan(sizes).any():
            raise ValueError(
                f"Missing segment sizes for lookahead segments {segment_number} to {segment_number + lookahead - 1}"
            )

        return np.array(sizes)

    def _kbps_to_bytes_per_s(self, kbps: float) -> float:
        return kbps * 1000 / 8

    def _bytes_per_s_to_kbps(self, bytes_per_s: float) -> float:
        return bytes_per_s / 1000

    def _get_action_from_entry(self, index: int, entry: BitrateLadderEntry) -> Action:
        return Action(
            bitrate_index=index,
            bitrate_kbps=entry["bitrate_kbps"],
            resolution_width=entry["resolution_width"],
            resolution_height=entry["resolution_height"],
            vmaf=entry["vmaf"],
            segment_size_bytes=entry["segment_size_bytes"],
        )
