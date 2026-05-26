from . import ABRPolicyClass

from ..Core.Interfaces import ABRPolicy
from ..Core.Types import Action, BitrateLadder, BitrateLadderEntry


import numpy as np


@ABRPolicyClass(name="WISH")
class WISHPolicy(ABRPolicy):
    def __init__(self):
        super().__init__()

        self.low_buffer_threshold_s = 5.0

        self.omega = 1.0 / 8.0
        self.mu = 0.1
        self.xi = 0.8
        self.delta = 1.0

        self.smoothed_throughput_bytes_per_second = None

    def select_action(self, state_t, ladder) -> Action:

        buffer_s = float(state_t.buffer_s)

        if buffer_s < self.low_buffer_threshold_s:
            return self._make_action(ladder, bitrate_index=0)

        if len(state_t.last_throughputs_kbps) == 0:
            return self._make_action(ladder, bitrate_index=0)
        else:
            last_throughput = state_t.last_throughputs_kbps[-1]
            smoothed_throughput_bytes_per_s = last_throughput * 1000

        if self.smoothed_throughput_bytes_per_second is None:
            self.smoothed_throughput_bytes_per_second = smoothed_throughput_bytes_per_s
        else:
            self.smoothed_throughput_bytes_per_s = (
                1.0 - self.omega
            ) * self.smoothed_throughput_bytes_per_s + self.omega * last_throughput

        estim_throughput_bytes_per_s = min(
            self.smoothed_throughput_bytes_per_s, smoothed_throughput_bytes_per_s
        )

        alpha, beta, gamma = self._compute_weights(state_t, ladder)

        q_recent = self._recent_quality_average(state_t, ladder)

        start_idx = 1 if len(ladder) > 1 else 0

        max_candidate_idx = self._max_candidate_index(
            ladder=ladder,
            last_throughput_bytes_per_s=smoothed_throughput_bytes_per_s,
            start_idx=start_idx,
        )

        # If no steady-phase candidate is sustainable, fall back to lowest.
        if max_candidate_idx < start_idx:
            return self._make_action(ladder, 0)

        best_idx = start_idx
        best_cost = float("inf")

        for idx in range(start_idx, max_candidate_idx + 1):
            entry = ladder.get_entry(idx)

            cost = self._wish_cost(
                entry=entry,
                ladder=ladder,
                estimated_throughput_bytes_per_s=estim_throughput_bytes_per_s,
                buffer_s=buffer_s,
                q_recent=q_recent,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        return self._make_action(ladder, best_idx)

    def _max_candidate_index(
        self,
        *,
        ladder,
        last_throughput_bytes_per_s: float,
        start_idx: int,
    ) -> int:
        max_idx = start_idx - 1
        budget_bytes_per_s = last_throughput_bytes_per_s * (1.0 + self.mu)

        for idx in range(start_idx, len(ladder)):
            entry = ladder.get_entry(idx)
            bitrate_bytes_per_s = self._bitrate_bytes_per_s(entry)

            if bitrate_bytes_per_s < budget_bytes_per_s:
                max_idx = idx
            else:
                break

        return max_idx

    def _compute_weights(self, state_t, ladder) -> tuple[float, float, float]:
        """
        WISH weight calculation.

        The paper uses:
            alpha + beta + gamma = 1
            alpha = 1 / (1 + BufferTerm + ExpTerm)
            beta = alpha * BufferTerm
            gamma = 1 - alpha - beta

        and derives alpha/gamma from the condition that the highest bitrate
        should be optimal when network/buffer/recent quality are favourable.
        """

        r_1 = float(ladder.get_entry(0)["bitrate_kbps"])
        r_n = float(ladder.get_entry(len(ladder) - 1)["bitrate_kbps"])

        q_1 = r_1 / max(r_n, 1e-9)

        # Paper default uses Q = q(R_N-1).
        if len(ladder) > 1:
            r_n_minus_1 = float(ladder.get_entry(len(ladder) - 2)["bitrate_kbps"])
            q_threshold = r_n_minus_1 / max(r_n, 1e-9)
        else:
            q_threshold = q_1

        delta = 1.0

        buffer_term = (
            self.xi * self.max_buffer_s - self.buffer_low_s
        ) / self.segment_duration_s

        exp_term = np.exp(3.0 - 2.0 * q_1 - q_threshold) / delta

        alpha = 1.0 / (1.0 + buffer_term + exp_term)
        beta = alpha * buffer_term
        gamma = 1.0 - alpha - beta

        return alpha, beta, gamma

    def _wish_cost(
        self,
        *,
        entry,
        ladder,
        estimated_throughput_bytes_per_s: float,
        buffer_s: float,
        q_recent: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> float:
        bitrate_bytes_per_s = self._bitrate_bytes_per_s(entry)
        estimated_throughput = max(estimated_throughput_bytes_per_s, 1e-9)

        segment_size_bytes = self._segment_size_bytes(entry)

        c_t = bitrate_bytes_per_s / estimated_throughput

        safe_buffer_s = max(buffer_s - self.buffer_low_s, 1e-9)
        estimated_download_time_s = segment_size_bytes / estimated_throughput
        c_b = estimated_download_time_s / safe_buffer_s

        # WISH quality cost, using bitrate-normalized q(i).
        c_q = self._quality_cost(
            entry=entry,
            ladder=ladder,
            q_recent=q_recent,
        )

        return alpha * c_t + beta * c_b + gamma * c_q

    def _recent_quality_average(self, state_t, ladder) -> float:
        """
        Q_k = average quality of recent selected representations.

        Preferred:
            state_t.bitrate_index_history

        Fallback:
            state_t.last_bitrate_index
        """
        bitrate_history = list(getattr(state_t, "bitrate_index_history", []))

        if len(bitrate_history) > 0:
            recent = bitrate_history[-self.quality_history_len :]

            qualities = [
                self._quality(ladder.get_entry(int(idx)), ladder)
                for idx in recent
                if idx is not None and 0 <= int(idx) < len(ladder)
            ]

            if len(qualities) > 0:
                return float(np.mean(qualities))

        last_idx = getattr(state_t, "last_bitrate_index", None)

        if last_idx is not None and 0 <= int(last_idx) < len(ladder):
            return self._quality(ladder.get_entry(int(last_idx)), ladder)

        return self._quality(ladder.get_entry(0), ladder)

    def _quality(self, entry, ladder) -> float:
        """
        Standard WISH quality:
            q(i) = R_i / R_N
        """
        bitrate_kbps = float(self._get(entry, "bitrate_kbps"))
        max_bitrate_kbps = float(
            self._get(ladder.get_entry(len(ladder) - 1), "bitrate_kbps")
        )

        return bitrate_kbps / max(max_bitrate_kbps, 1e-9)

    def _get_throughput_history(self, state_t) -> list[float]:
        history = list(getattr(state_t, "last_throughputs_kbps", []))

        if len(history) > 0:
            return [
                float(x) * 1000 for x in history if x is not None and float(x) > 0.0
            ]

        last = getattr(state_t, "last_throughput_bytes_per_s", None)

        if last is None:
            return []

        last = float(last)

        if last <= 0.0:
            return []

        return [last]

    def _get(self, entry, key: str, default=None):
        if isinstance(entry, dict):
            return entry.get(key, default)

        return getattr(entry, key, default)

    def _make_action(self, ladder: BitrateLadder, bitrate_index) -> Action:
        entry: BitrateLadderEntry = ladder.get_entry(bitrate_index)

        return Action(
            bitrate_index=bitrate_index,
            bitrate_kbps=entry["bitrate_kbps"],
            resolution_width=entry["resolution_width"],
            resolution_height=entry["resolution_height"],
            vmaf=entry["vmaf"],
            segment_size_bytes=entry["segment_size_bytes"],
        )
