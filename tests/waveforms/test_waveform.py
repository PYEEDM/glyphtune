"""Tests for the base waveform class."""

from typing import override
import math
import numpy as np
import pytest
from glyphtune import signal, waveforms


class TimeWaveform(waveforms.Waveform):
    """Useless waveform that just returns the time signal when sampled."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return time


def test_sample_seconds_creates_time_signal_with_correct_shape() -> None:
    """Ensure `sample_seconds` creates time signal with correct number of channels and samples."""
    sampling_rate = 10
    duration = 1.4
    channels = 5
    time_waveform = TimeWaveform()

    time = time_waveform.sample_seconds(duration, sampling_rate, channels=channels)

    assert time.shape == (channels, math.ceil(sampling_rate * duration))


def test_sample_seconds_creates_evenly_spaced_time_signal() -> None:
    """Ensure `sample_seconds` creates an evenly spaced time signal."""
    sampling_rate = 10
    duration = 1.4
    time_waveform = TimeWaveform()

    time = time_waveform.sample_seconds(duration, sampling_rate, channels=1)

    time_differences = np.diff(np.squeeze(time))
    assert time_differences == pytest.approx(time_differences[0])


def test_sample_seconds_time_signal_first_value_equals_offset() -> None:
    """Ensure first value of time signal created by `sample_seconds` is `start_offset`."""
    sampling_rate = 10
    duration = 1.4
    start_offset = 2.6
    time_waveform = TimeWaveform()

    time = time_waveform.sample_seconds(
        duration, sampling_rate, start_offset, channels=1
    )

    assert time[0, 0] == start_offset


def test_sample_samples_creates_time_signal_with_correct_shape() -> None:
    """Ensure `sample_samples` creates time signal with correct number of channels and samples."""
    sampling_rate = 7
    count = 10
    channels = 2
    time_waveform = TimeWaveform()

    time = time_waveform.sample_samples(count, sampling_rate, channels=channels)

    assert time.shape == (channels, count)
