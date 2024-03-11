"""Tests for the base waveform class."""

from typing import override
import math
import numpy as np
import pytest
from glyphtune import arrays, waveforms


class TimeArrayWaveform(waveforms.Waveform):
    """Useless waveform that just returns the time array when sampled."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        return time_array


def test_sample_seconds_creates_time_array_with_correct_shape() -> None:
    """Ensure `sample_seconds` creates time array with correct number of channels and samples."""
    sampling_rate = 10
    duration = 1.4
    channels = 5
    time_array_waveform = TimeArrayWaveform()

    time_array = time_array_waveform.sample_seconds(
        sampling_rate, duration, channels=channels
    )

    assert time_array.shape == (channels, math.ceil(sampling_rate * duration))


def test_sample_seconds_creates_evenly_spaced_time_array() -> None:
    """Ensure `sample_seconds` creates an evenly spaced time array."""
    sampling_rate = 10
    duration = 1.4
    time_array_waveform = TimeArrayWaveform()

    time_array = time_array_waveform.sample_seconds(sampling_rate, duration, channels=1)

    time_array_differences = np.diff(time_array.squeeze())
    assert time_array_differences == pytest.approx(time_array_differences[0])


def test_sample_seconds_time_array_first_value_equals_offset() -> None:
    """Ensure first value of time array created by `sample_seconds` is `start_offset`."""
    sampling_rate = 10
    duration = 1.4
    start_offset = 2.6
    time_array_waveform = TimeArrayWaveform()

    time_array = time_array_waveform.sample_seconds(
        sampling_rate, duration, start_offset, channels=1
    )

    assert time_array[0, 0] == start_offset


def test_sample_samples_creates_time_array_with_correct_shape() -> None:
    """Ensure `sample_samples` creates a time array with correct number of channels and samples."""
    sampling_rate = 7
    count = 10
    channels = 2
    time_array_waveform = TimeArrayWaveform()

    time_array = time_array_waveform.sample_samples(
        sampling_rate, count, channels=channels
    )

    assert time_array.shape == (channels, count)
