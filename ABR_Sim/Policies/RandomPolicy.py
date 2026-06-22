from random import Random
from . import ABRPolicyClass
from ..Core.Interfaces import ABRPolicy
from ..Core.Types import ScenarioConfig, Action, BitrateLadderEntry


@ABRPolicyClass(name="Random")
class RandomPolicy(ABRPolicy):

    def __init__(self, scenario_config: ScenarioConfig, seed: int = None):
        super().__init__(scenario_config=scenario_config, seed=seed)
        self.random = Random(seed)

    def select_action(self, state_t, ladder) -> Action:
        num_representations = len(ladder)

        ladder_entry: BitrateLadderEntry = ladder.get_entry(
            self.random.randint(0, num_representations - 1)
        )

        return Action(
            bitrate_index=ladder.entries.index(ladder_entry),
            bitrate_kbps=ladder_entry["bitrate_kbps"],
            resolution_width=ladder_entry["resolution_width"],
            resolution_height=ladder_entry["resolution_height"],
            vmaf=ladder_entry["vmaf"],
            segment_size_bytes=ladder_entry["segment_size_bytes"],
        )
