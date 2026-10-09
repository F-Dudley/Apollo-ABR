from typing import Callable

from . import ABRPolicyClass
from ..Core.Interfaces import ABRPolicy
from ..Core.Types import BitrateLadder, ScenarioConfig, Action

from random import Random


@ABRPolicyClass(name="RandomWalk")
class RandomWalkPolicy(ABRPolicy):

    def __init__(
        self, scenario_config: ScenarioConfig, seed: int = None, max_step_size: int = 1
    ):
        super().__init__(scenario_config=scenario_config, seed=seed)

        self.rng = Random(seed)
        self.max_step_size = max_step_size

    def select_action(
        self,
        state_t,
        ladder,
        *,
        segment_number: int,
        max_segment_number: int,
        segment_lookup: Callable[[int], BitrateLadder],
    ) -> Action:
        num_representations = len(ladder)

        if state_t.last_actions is None or len(state_t.last_actions) == 0:
            new_idx = self.rng.randint(0, num_representations - 1)
        else:
            new_idx = self._get_stepped_bitrate_index(
                state_t.last_actions, num_representations
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

    def _get_stepped_bitrate_index(self, last_actions, num_representations) -> int:
        if last_actions is None or len(last_actions) == 0:
            new_idx = self.rng.randint(0, num_representations - 1)
        else:
            delta = self.rng.randint(-self.max_step_size, self.max_step_size)
            new_idx = max(
                0, min(last_actions[-1].bitrate_index + delta, num_representations - 1)
            )
        return new_idx
