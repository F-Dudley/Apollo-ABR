from __future__ import annotations

import math
from typing import Callable

from . import ABRPolicyClass
from ..Core.Interfaces import ABRPolicy
from ..Core.Types import (
    BitrateLadder,
    ScenarioConfig,
    Action,
)


@ABRPolicyClass(name="BOLA")
class BOLAPolicy(ABRPolicy):
    """
    BOLA implementation aligned with dash.js 5.2.0 for validation.

    Matches the configured DASH.js setup:
      - BOLA enabled
      - ThroughputRule disabled
      - InsufficientBufferRule disabled
      - AbandonRequestsRule disabled
      - Fast switching disabled
      - 30 second configured buffer target

    BOLA still internally uses safe EWMA throughput:
      - during startup
      - for BOLA-O up-switch suppression
    """

    MINIMUM_BUFFER_S = 10.0
    MINIMUM_BUFFER_PER_BITRATE_LEVEL_S = 2.0

    PLACEHOLDER_BUFFER_DECAY = 0.99

    # dash.js 5.2 defaults
    EWMA_FAST_HALF_LIFE_S = 3.0
    EWMA_SLOW_HALF_LIFE_S = 8.0
    EWMA_WEIGHT_DOWNLOAD_TIME_FACTOR = 0.0015
    BANDWIDTH_SAFETY_FACTOR = 0.9

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        seed: int | None = None,
    ):
        super().__init__(
            scenario_config=scenario_config,
            seed=seed,
        )

        # -------------------------------------------------------------
        # BOLA state
        # -------------------------------------------------------------

        self._initialized = False
        self._startup = True

        self._bitrates_kbps: list[float] = []
        self._utilities: list[float] = []

        self._vp = 0.0
        self._gp = 0.0

        self._current_idx: int | None = None

        self._placeholder_buffer_s = 0.0

        # -------------------------------------------------------------
        # EWMA throughput state
        # -------------------------------------------------------------

        self._ewma_fast = 0.0
        self._ewma_slow = 0.0
        self._ewma_total_weight = 0.0

        # Prevent the same completed segment being fed into EWMA twice.
        self._last_observed_segment: int | None = None

        # Can be consumed by the simulator later if you decide to
        # reproduce BOLA's scheduling delay.
        self.last_scheduling_delay_s = 0.0

        self._max_buffer_s = float(
            getattr(
                scenario_config,
                "max_buffer_s",
                30.0,
            )
        )

    def select_action(
        self,
        state_t,
        ladder: BitrateLadder,
        *,
        segment_number: int,
        max_segment_number: int,
        segment_lookup: Callable[
            [int],
            BitrateLadder,
        ],
    ) -> Action:

        del max_segment_number
        del segment_lookup

        if not self._initialized:
            self._initialize_bola(ladder)

        # The state passed for segment N contains the result of
        # downloading segment N-1. Feed that completed request into
        # the EWMA estimator.
        self._observe_previous_download(
            state_t=state_t,
            segment_number=segment_number,
        )

        # Only one representation.
        if len(self._bitrates_kbps) == 1:
            return self._make_action(
                ladder=ladder,
                index=0,
            )

        buffer_s = float(state_t.buffer_s)

        # Equivalent to dash.js BUFFER_EMPTY behaviour.
        if buffer_s <= 0.0:
            self._placeholder_buffer_s = 0.0

        safe_throughput_kbps = self._get_safe_ewma_throughput_kbps()

        # -------------------------------------------------------------
        # Initial request
        # -------------------------------------------------------------

        if state_t.last_actions is None or len(state_t.last_actions) == 0:
            #
            # Keep the same deterministic startup behaviour you
            # already had.
            #
            # DASH.js may not yet have a throughput measurement for
            # BOLA on the very first request either.
            #
            best_idx = 0
            self._current_idx = best_idx

            return self._make_action(
                ladder=ladder,
                index=best_idx,
            )

        # -------------------------------------------------------------
        # BOLA startup
        # -------------------------------------------------------------

        if self._startup:
            best_idx = self._select_startup(
                buffer_s=buffer_s,
                safe_throughput_kbps=(safe_throughput_kbps),
            )

            segment_duration_s = float(state_t.config.segment_duration_s)

            # dash.js switches into steady BOLA once at least one
            # segment has completed and sufficient media is buffered.
            if buffer_s >= segment_duration_s:
                self._startup = False

        # -------------------------------------------------------------
        # BOLA steady state
        # -------------------------------------------------------------

        else:
            best_idx = self._select_steady(
                buffer_s=buffer_s,
                safe_throughput_kbps=(safe_throughput_kbps),
            )

        self._current_idx = best_idx

        return self._make_action(
            ladder=ladder,
            index=best_idx,
        )

    # =================================================================
    # BOLA initialisation
    # =================================================================

    def _initialize_bola(
        self,
        ladder: BitrateLadder,
    ) -> None:

        self._bitrates_kbps = [
            float(ladder.get_entry(idx)["bitrate_kbps"]) for idx in range(len(ladder))
        ]

        if len(self._bitrates_kbps) == 0:
            raise ValueError("BOLA requires at least one representation.")

        if any(bitrate <= 0.0 for bitrate in self._bitrates_kbps):
            raise ValueError("BOLA bitrates must be positive.")

        # DASH.js assumes representations are ordered from lowest to
        # highest bandwidth.
        if any(
            self._bitrates_kbps[i] > self._bitrates_kbps[i + 1]
            for i in range(len(self._bitrates_kbps) - 1)
        ):
            raise ValueError(
                "BOLA ladder must be ordered from " "lowest to highest bitrate."
            )

        # dash.js:
        #
        # utilities = bitrates.map(Math.log)
        # utilities =
        #     utilities.map(u => u - utilities[0] + 1)
        #
        raw_utilities = [math.log(bitrate) for bitrate in self._bitrates_kbps]

        minimum_utility = raw_utilities[0]

        self._utilities = [utility - minimum_utility + 1.0 for utility in raw_utilities]

        if len(self._bitrates_kbps) > 1:
            self._calculate_bola_parameters()

        self._initialized = True

    def _calculate_bola_parameters(
        self,
    ) -> None:

        highest_utility = max(self._utilities)

        # Your DASH.js configuration uses bufferTimeDefault = 30.
        configured_buffer_s = self._max_buffer_s

        # dash.js does not necessarily use exactly 30 seconds for
        # BOLA's internal parameters.
        #
        # bufferTime =
        #   max(
        #       bufferTimeDefault,
        #       10 + 2 * representationCount
        #   )
        #
        buffer_time_s = max(
            configured_buffer_s,
            (
                self.MINIMUM_BUFFER_S
                + (self.MINIMUM_BUFFER_PER_BITRATE_LEVEL_S * len(self._bitrates_kbps))
            ),
        )

        self._gp = (highest_utility - 1.0) / (
            buffer_time_s / self.MINIMUM_BUFFER_S - 1.0
        )

        self._vp = self.MINIMUM_BUFFER_S / self._gp

    # =================================================================
    # Startup
    # =================================================================

    def _select_startup(
        self,
        *,
        buffer_s: float,
        safe_throughput_kbps: float | None,
    ) -> int:

        # Until DASH.js has a throughput measurement it does not have
        # a new BOLA startup recommendation.
        if safe_throughput_kbps is None:
            return self._current_idx if self._current_idx is not None else 0

        best_idx = self._optimal_index_for_throughput(safe_throughput_kbps)

        # dash.js sets placeholder buffer so that entering steady BOLA
        # does not immediately disagree with the throughput-selected
        # startup representation.
        min_buffer_s = self._min_buffer_level_for_representation(best_idx)

        self._placeholder_buffer_s = max(
            0.0,
            min_buffer_s - buffer_s,
        )

        self.last_scheduling_delay_s = 0.0

        return best_idx

    def _select_steady(
        self,
        *,
        buffer_s: float,
        safe_throughput_kbps: float | None,
    ) -> int:

        effective_buffer_s = buffer_s + self._placeholder_buffer_s

        best_idx = self._representation_from_buffer(effective_buffer_s)

        # -------------------------------------------------------------
        # BOLA-O
        # -------------------------------------------------------------
        if safe_throughput_kbps is not None and self._current_idx is not None:
            throughput_idx = self._optimal_index_for_throughput(safe_throughput_kbps)

            if best_idx > self._current_idx and best_idx > throughput_idx:
                best_idx = (
                    throughput_idx
                    if throughput_idx > self._current_idx
                    else self._current_idx
                )

        # Scheduling delay
        delay_s = max(
            0.0,
            (
                buffer_s
                + self._placeholder_buffer_s
                - self._max_buffer_level_for_representation(best_idx)
            ),
        )

        if delay_s <= self._placeholder_buffer_s:
            self._placeholder_buffer_s -= delay_s
            delay_s = 0.0

        else:
            delay_s -= self._placeholder_buffer_s
            self._placeholder_buffer_s = 0.0

            # dash.js leaves top-quality buffering to its normal
            # scheduling/buffer controller.
            if best_idx == len(self._bitrates_kbps) - 1:
                delay_s = 0.0

        self.last_scheduling_delay_s = delay_s

        return best_idx

    def _representation_from_buffer(
        self,
        buffer_s: float,
    ) -> int:

        best_idx = 0
        best_score = float("-inf")

        for idx, bitrate_kbps in enumerate(self._bitrates_kbps):
            score = (
                self._vp * (self._utilities[idx] - 1.0 + self._gp) - buffer_s
            ) / bitrate_kbps

            if score >= best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _min_buffer_level_for_representation(
        self,
        idx: int,
    ) -> float:

        bitrate = self._bitrates_kbps[idx]
        utility = self._utilities[idx]

        minimum_buffer_s = 0.0

        for lower_idx in range(
            idx - 1,
            -1,
            -1,
        ):
            lower_utility = self._utilities[lower_idx]

            if lower_utility >= utility:
                continue

            lower_bitrate = self._bitrates_kbps[lower_idx]

            level = self._vp * (
                self._gp
                + ((bitrate * lower_utility) - (lower_bitrate * utility))
                / (bitrate - lower_bitrate)
            )

            minimum_buffer_s = max(
                minimum_buffer_s,
                level,
            )

        return minimum_buffer_s

    def _max_buffer_level_for_representation(
        self,
        idx: int,
    ) -> float:

        return self._vp * (self._utilities[idx] + self._gp)

    # =================================================================
    # DASH.js-style EWMA
    # =================================================================

    def _observe_previous_download(
        self,
        *,
        state_t,
        segment_number: int,
    ) -> None:
        """
        Add exactly one completed media request to the throughput EWMA.

        This should occur once per completed segment, NOT once per
        one-second network-trace sample.
        """

        previous_segment = segment_number - 1

        if previous_segment < 0:
            return

        if self._last_observed_segment == previous_segment:
            return

        if state_t.last_actions is None or len(state_t.last_actions) == 0:
            return

        download_time_s = self._get_download_time_s(state_t)

        if download_time_s is None or download_time_s <= 0.0:
            return

        previous_action = state_t.last_actions[-1]

        segment_size_bytes = self._get_action_value(
            previous_action,
            "segment_size_bytes",
        )

        if segment_size_bytes is None or segment_size_bytes <= 0:
            return

        throughput_kbps = round(
            (8.0 * float(segment_size_bytes)) / (download_time_s * 1000.0)
        )

        self._update_ewma(
            throughput_kbps=throughput_kbps,
            download_time_s=download_time_s,
        )

        self._last_observed_segment = previous_segment

        # dash.js decays placeholder buffer whenever a new completed
        # segment is observed.
        self._placeholder_buffer_s *= self.PLACEHOLDER_BUFFER_DECAY

    def _update_ewma(
        self,
        *,
        throughput_kbps: float,
        download_time_s: float,
    ) -> None:

        weight = self.EWMA_WEIGHT_DOWNLOAD_TIME_FACTOR * download_time_s * 1000.0

        fast_alpha = math.pow(
            0.5,
            weight / self.EWMA_FAST_HALF_LIFE_S,
        )

        slow_alpha = math.pow(
            0.5,
            weight / self.EWMA_SLOW_HALF_LIFE_S,
        )

        self._ewma_fast = (
            1.0 - fast_alpha
        ) * throughput_kbps + fast_alpha * self._ewma_fast

        self._ewma_slow = (
            1.0 - slow_alpha
        ) * throughput_kbps + slow_alpha * self._ewma_slow

        self._ewma_total_weight += weight

    def _get_safe_ewma_throughput_kbps(
        self,
    ) -> float | None:

        if self._ewma_total_weight <= 0.0:
            return None

        # dash.js startup correction.
        fast_zero_factor = 1.0 - math.pow(
            0.5,
            self._ewma_total_weight / self.EWMA_FAST_HALF_LIFE_S,
        )

        slow_zero_factor = 1.0 - math.pow(
            0.5,
            self._ewma_total_weight / self.EWMA_SLOW_HALF_LIFE_S,
        )

        fast_estimate = self._ewma_fast / fast_zero_factor

        slow_estimate = self._ewma_slow / slow_zero_factor

        # For bandwidth dash.js selects the minimum of the fast and slow EWMA estimates.
        average_kbps = min(
            fast_estimate,
            slow_estimate,
        )

        average_kbps = round(average_kbps)

        return average_kbps * self.BANDWIDTH_SAFETY_FACTOR

    # =================================================================
    # Throughput quality selection
    # =================================================================

    def _optimal_index_for_throughput(
        self,
        throughput_kbps: float,
    ) -> int:

        best_idx = 0

        for idx, bitrate_kbps in enumerate(self._bitrates_kbps):
            if bitrate_kbps <= throughput_kbps:
                best_idx = idx
            else:
                break

        return best_idx

    # =================================================================
    # State helpers
    # =================================================================

    @staticmethod
    def _get_download_time_s(
        state_t,
    ) -> float | None:
        """
        Adapt these names to the exact State type if required.

        Ideally this is the measured duration of the PREVIOUS segment
        download, excluding request latency because your DASH.js
        configuration has useDeadTimeLatency=true.
        """
        value = getattr(
            state_t,
            "last_download_time_s",
            None,
        )

        if value is not None:
            return float(value)

        return None

    @staticmethod
    def _get_action_value(
        action,
        key: str,
    ):
        if isinstance(action, dict):
            return action.get(key)

        return getattr(
            action,
            key,
            None,
        )

    # =================================================================
    # Action
    # =================================================================

    @staticmethod
    def _make_action(
        *,
        ladder: BitrateLadder,
        index: int,
    ) -> Action:

        entry = ladder.get_entry(index)

        return Action(
            bitrate_index=index,
            bitrate_kbps=entry["bitrate_kbps"],
            resolution_width=entry["resolution_width"],
            resolution_height=entry["resolution_height"],
            vmaf=entry["vmaf"],
            segment_size_bytes=entry["segment_size_bytes"],
        )
