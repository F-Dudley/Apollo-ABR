from dataclasses import dataclass
from .Core.Types import Transition

from tqdm.auto import tqdm


@dataclass(frozen=True)
class WindowConfig:
    context_size: int = 4
    target_size: int = 1


@dataclass(frozen=True)
class TemporalWindow:
    scenario_id: str
    anchor_step_t: int

    context: list[Transition]
    target: list[Transition]


class TemporalWindowBuilder:
    def __init__(self, config: WindowConfig, show_progress: bool = False):
        self.config = config
        self.show_progress = show_progress

    def build(self, transitions: list[Transition]) -> list[TemporalWindow]:

        windows: list[TemporalWindow] = []

        total_size = self.config.context_size + self.config.target_size

        if len(transitions) < total_size:
            raise ValueError(
                f"Not enough transitions to build a single window. Required: {total_size}, provided: {len(transitions)}"
            )

        num_windows = len(transitions) - total_size + 1
        iterator = range(num_windows)

        if self.show_progress:
            iterator = tqdm(
                iterator,
                desc="Building temporal windows...",
                total=num_windows,
                unit="Window",
                leave=False,
            )

        for start_idx in iterator:

            context_start_idx = start_idx
            context_end_idx = start_idx + self.config.context_size

            target_start_idx = context_end_idx
            target_end_idx = target_start_idx + self.config.target_size

            window = TemporalWindow(
                scenario_id=transitions[start_idx].scenario_id,
                anchor_step_t=transitions[context_end_idx - 1].step_t,
                context=transitions[context_start_idx:context_end_idx],
                target=transitions[target_start_idx:target_end_idx],
            )
            windows.append(window)

        return windows
