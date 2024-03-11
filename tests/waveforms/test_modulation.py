"""Tests for modulation waveforms."""

from typing import override
import pytest
from glyphtune import arrays, waveforms


class TimeArrayPeriodicWave(waveforms.PeriodicWave):
    """Useless waveform that just returns the time array when sampled."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        return time_array


def test_phase_modulation_shifts_time_array_by_modulator() -> None:
    """Ensure that phase modulating a wave passes to it a time array shifted by the modulator."""
    sampling_rate = 10
    duration = 1
    time_array_waveform = TimeArrayPeriodicWave(1)
    modulator = waveforms.Sine(1)

    phase_modulated_waveform = waveforms.phase_modulate(time_array_waveform, modulator)

    sampled_modulated_waveform = phase_modulated_waveform.sample_seconds(
        sampling_rate, duration
    )
    sampled_time_array = time_array_waveform.sample_seconds(sampling_rate, duration)
    sampled_modulator = modulator.sample_seconds(sampling_rate, duration)
    reference_samples = sampled_time_array + sampled_modulator
    assert sampled_modulated_waveform == pytest.approx(reference_samples)
