from . import ABRPolicyClass
from ..Core.Interfaces import ABRPolicy
from ..Core.Types import Action

import numpy as np


@ABRPolicyClass(name="Throughput")
class ThroughputPolicy(ABRPolicy):

    name = "Throughput"

    def __init__(self):
        super().__init__()

        self.history_size = 5
        self.safety_factor = 0.9

    def select_action(self, state_t, ladder) -> Action:
        if state_t.last_action is None:
            best_idx = 0
        else:
            estimated_bytes_per_second = self._harmonic_mean(
                state_t.last_throughputs_kbps * 1000
            )

            estimated_bytes_per_second *= self.safety_factor

            best_idx = 0
            for entry in ladder:

                bitrate_bytes_per_second = entry["bitrate_kbps"] * 1000

                if bitrate_bytes_per_second <= estimated_bytes_per_second:
                    best_idx += 1
                else:
                    break

        new_idx_entry = ladder.get_entry(best_idx)

        return Action(
            bitrate_index=best_idx,
            bitrate_kbps=new_idx_entry["bitrate_kbps"],
            resolution_width=new_idx_entry["resolution_width"],
            resolution_height=new_idx_entry["resolution_height"],
            vmaf=new_idx_entry["vmaf"],
            segment_size_bytes=new_idx_entry["segment_size_bytes"],
        )

    def _harmonic_mean(self, throughputs_kbps: list[float]) -> float:
        if len(throughputs_kbps) == 0:
            return 0.0

        arr = np.array(throughputs_kbps, dtype=float)
        arr = arr[
            arr > 0
        ]  # Filter out zero or negative throughputs to avoid division by zero

        if len(arr) == 0:
            return 0.0

        return float(len(arr) / np.sum(1.0 / arr))
