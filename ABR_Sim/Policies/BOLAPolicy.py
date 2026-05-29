from . import ABRPolicyClass
from ..Core.Interfaces import ABRPolicy
from ..Core.Types import Action, BitrateLadderEntry, get_ladder_entry


@ABRPolicyClass(name="BOLA")
class BOLAPolicy(ABRPolicy):

    def __init__(self):
        super().__init__()

        self.v_quality = 20.0
        self.gamma = 0.1

    def select_action(self, state_t, ladder) -> Action:
        if len(state_t.last_actions) == 0:
            best_idx = 0
        else:

            segment_duration_s = state_t.config.segment_duration_s

            best_cost = float("-inf")
            best_idx = 0
            for idx in range(len(ladder)):
                entry = ladder.get_entry(idx)
                cost = self._bola_cost(
                    quality=entry["vmaf"],
                    segment_size_bytes=entry["segment_size_bytes"],
                    segment_duration_s=segment_duration_s,
                    buffer_s=state_t.buffer_s,
                )
                if cost > best_cost:
                    best_cost = cost
                    best_idx = idx

        new_idx_entry = get_ladder_entry(ladder, best_idx)

        return Action(
            bitrate_index=best_idx,
            bitrate_kbps=new_idx_entry["bitrate_kbps"],
            resolution_width=new_idx_entry["resolution_width"],
            resolution_height=new_idx_entry["resolution_height"],
            vmaf=new_idx_entry["vmaf"],
            segment_size_bytes=new_idx_entry["segment_size_bytes"],
        )

    def _bola_cost(
        self,
        vmaf: float,
        segment_size_bytes: int,
        segment_duration_s: float,
        buffer_s: float,
    ):

        utility = self._normalized_vmaf(vmaf)

        return (
            self.v_quality * (utility + self.gamma * segment_duration_s) - buffer_s
        ) / (segment_size_bytes)

    def _normalized_vmaf(self, vmaf: float) -> float:
        return vmaf / 100.0
