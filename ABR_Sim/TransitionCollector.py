from .Core.Types import Transition
from .Simulator import ABRSimulator

from tqdm.auto import tqdm


class TransitionCollector:
    def __init__(
        self,
        simulator: ABRSimulator,
        show_progress: bool = False,
        progress_desc: str = "Simulating ABR scenario...",
    ):
        self.simulator = simulator
        self.show_progress = show_progress
        self.progress_desc = progress_desc

    def run(self) -> tuple[list[Transition], list[str]]:
        self.simulator.reset()

        transitions: list[Transition] = []

        iterator = range(self.simulator.num_segments())  # type: ignore

        if self.show_progress:
            iterator = tqdm(
                iterator,
                desc=self.progress_desc or "Simulating ABR Transitions",
                total=self.simulator.total_segments,
                unit="Segment",
                leave=False,
            )

        for _ in iterator:

            transition = self.simulator.step()
            transitions.append(transition)

            if transition.done:
                break

        try:
            transition_columns = list(transitions[0].__dict__.keys())
        except Exception:
            transition_columns = list(Transition.__dict__.keys())

        return transitions, transition_columns
