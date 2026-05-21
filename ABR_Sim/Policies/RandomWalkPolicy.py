from ..Core.Interfaces import ABRPolicy
from ..Core.Types import Action

from random import Random


class RandomWalkPolicy(ABRPolicy):

    def __init__(self, seed: int = None, max_step_size: int = 1):
        super().__init__()

        self.rng = Random(seed)
        self.max_step_size = max_step_size

    def select_action(self, state_t, ladder) -> Action:
        num_representations = len(ladder)

        new_idx = self._get_stepped_bitrate_index(
            state_t.last_action, num_representations
        )

        new_idx_entry = ladder.get_entry(new_idx)

        return Action(
            bitrate_index=new_idx,
            bitrate_kbps=new_idx_entry["bitrate_kbps"],
            resolution_width=new_idx_entry["resolution_width"],
            resolution_height=new_idx_entry["resolution_height"],
            vmaf=new_idx_entry["vmaf"],
            segment_size_bytes=new_idx_entry["segment_size_bytes"],
        )

    def _get_stepped_bitrate_index(self, action, num_representations) -> int:
        if action is None:
            new_idx = self.rng.randint(0, num_representations - 1)
        else:
            delta = self.rng.randint(-self.max_step_size, self.max_step_size)
            new_idx = max(0, min(action.bitrate_index + delta, num_representations - 1))
        return new_idx
