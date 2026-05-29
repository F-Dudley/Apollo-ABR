from dataclasses import dataclass


@dataclass(frozen=True)
class BufferPreDownload:
    buffer_s_before_wait: float
    wait_time_s: float
    buffer_s_before_download: float
    download_start_time_s: float


@dataclass(frozen=True)
class BufferPostDownload:
    download_time_s: float
    decoding_time_s: float
    segment_ready_time_s: float

    rebuffer_time_s: float

    buffer_s_after_download: float
    buffer_s_after_segment_ready: float
    buffer_s_next: float

    download_end_time_s: float
    segment_ready_at_s: float
    total_time_used_s: float


class BufferManager:

    def __init__(
        self,
        segment_duration_s: float,
        max_buffer_s: float,
        wait_for_space: bool = True,
        include_decoding_time: bool = False,
    ):

        self.segment_duration_s = segment_duration_s
        self.max_buffer_s = max_buffer_s
        self.wait_for_space = wait_for_space
        self.include_decoding_time = include_decoding_time

        self.buffer_threshold_s = max_buffer_s - segment_duration_s

    def prepare_download(self, buffer_s: float, sim_time_s: float) -> BufferPreDownload:
        buffer_s_before_wait = buffer_s
        wait_time_s = 0.0

        if self.wait_for_space and buffer_s > self.buffer_threshold_s:
            wait_time_s = max(0.0, buffer_s - self.buffer_threshold_s)
            buffer_s -= wait_time_s
            sim_time_s += wait_time_s

        return BufferPreDownload(
            buffer_s_before_wait=buffer_s_before_wait,
            wait_time_s=wait_time_s,
            buffer_s_before_download=buffer_s,
            download_start_time_s=sim_time_s,
        )

    def complete_download(
        self,
        buffer_s_before_download: float,
        download_start_time_s: float,
        download_time_s: float,
        decoding_time_s: float = 0.0,
        wait_time_s: float = 0.0,
    ) -> BufferPostDownload:

        if not self.include_decoding_time:
            decoding_time_s = 0.0
        else:
            decoding_time_s = max(0.0, decoding_time_s)

        wait_time_s = max(0.0, wait_time_s)
        download_time_s = max(0.0, download_time_s)

        download_end_time_s = download_start_time_s + download_time_s

        segment_ready_delta_s = download_time_s + decoding_time_s
        segment_ready_at_s = download_start_time_s + segment_ready_delta_s

        rebuffer_time_s = max(0.0, segment_ready_delta_s - buffer_s_before_download)

        buffer_s_after_download = max(0.0, buffer_s_before_download - download_time_s)
        buffer_s_after_segment_ready = max(
            0.0, buffer_s_before_download - segment_ready_delta_s
        )

        buffer_s_next = min(
            buffer_s_after_segment_ready + self.segment_duration_s, self.max_buffer_s
        )

        total_time_used_s = wait_time_s + segment_ready_delta_s

        # Validation Checks
        expected_rebuffer_time_s = max(
            0.0,
            segment_ready_delta_s - buffer_s_before_download,
        )

        expected_total_time_used_s = wait_time_s + segment_ready_delta_s

        assert abs(rebuffer_time_s - expected_rebuffer_time_s) < 1e-9, (
            f"Bad rebuffer calculation: got {rebuffer_time_s}, "
            f"expected {expected_rebuffer_time_s}. "
            f"buffer={buffer_s_before_download}, "
            f"download={download_time_s}, "
            f"decode={decoding_time_s}, "
            f"segment_ready_delta={segment_ready_delta_s}"
        )

        assert abs(total_time_used_s - expected_total_time_used_s) < 1e-9, (
            f"Bad total_time_used_s: got {total_time_used_s}, "
            f"expected {expected_total_time_used_s}"
        )

        return BufferPostDownload(
            download_time_s=download_time_s,
            decoding_time_s=decoding_time_s,
            segment_ready_time_s=segment_ready_delta_s,
            rebuffer_time_s=rebuffer_time_s,
            buffer_s_after_download=buffer_s_after_download,
            buffer_s_after_segment_ready=buffer_s_after_segment_ready,
            buffer_s_next=buffer_s_next,
            segment_ready_at_s=segment_ready_at_s,
            download_end_time_s=download_end_time_s,
            total_time_used_s=total_time_used_s,
        )
