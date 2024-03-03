"""Tests for modulation waveforms."""

from typing import override
import pytest
import glyphtune
from glyphtune import waveforms


class TimeArrayPeriodicWave(waveforms.PeriodicWave):
    """Useless waveform that just returns the time array when sampled."""

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
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


def test_phase_modulation_by_sine_equals_frequency_modulation_by_cosine() -> None:
    """Ensure phase modulating by sine modulator equals frequency modulating by cosine modulator."""
    sampling_rate = 10**3
    duration = 1
    carrier = waveforms.Sine(20)
    sine_modulator = waveforms.Sine(1)
    cosine_modulator = waveforms.Sine(1, 0.25)
    error_margin = 10**-2

    phase_modulated_waveform = waveforms.phase_modulate(carrier, sine_modulator)
    frequency_modulated_waveform = waveforms.frequency_modulate(
        carrier, cosine_modulator
    )

    sampled_phase_modulation = phase_modulated_waveform.sample_seconds(
        sampling_rate, duration
    )
    sampled_frequency_modulation = frequency_modulated_waveform.sample_seconds(
        sampling_rate, duration
    )
    assert sampled_phase_modulation == pytest.approx(
        sampled_frequency_modulation, abs=error_margin
    )
