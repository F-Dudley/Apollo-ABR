from dataclasses import dataclass


@dataclass(frozen=True)
class BufferPreDownload:
    buffer_s_before_wait: float
    wait_time_s: float
    buffer_s_before_download: float
    download_start_time_s: float


@dataclass(frozen=True)
class BufferPostDownload:
    rebuffer_time_s: float
    buffer_s_after_download: float
    buffer_s_next: float
    download_end_time_s: float


class BufferManager:

    def __init__(
        self,
        segment_duration_s: float,
        max_buffer_s: float,
        wait_for_space: bool = True,
    ):

        self.segment_duration_s = segment_duration_s
        self.max_buffer_s = max_buffer_s

        self.buffer_threshold_s = max_buffer_s - segment_duration_s
        self.wait_for_space = wait_for_space

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
    ) -> BufferPostDownload:

        rebuffer_time_s = max(0.0, download_time_s - buffer_s_before_download)

        buffer_s_after_download = max(0.0, buffer_s_before_download - download_time_s)
        buffer_s_next = min(
            buffer_s_after_download + self.segment_duration_s, self.max_buffer_s
        )

        download_end_time_s = download_start_time_s + download_time_s

        return BufferPostDownload(
            rebuffer_time_s=rebuffer_time_s,
            buffer_s_after_download=buffer_s_after_download,
            buffer_s_next=buffer_s_next,
            download_end_time_s=download_end_time_s,
        )
