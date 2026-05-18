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

        if not self.simulator.initialized:
            self.simulator.init()

    def run(self) -> list[Transition]:

        if not self.simulator.initialized:
            self.simulator.init()

        transitions: list[Transition] = []

        iterator = range(self.simulator.total_segments)

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

        return transitions
