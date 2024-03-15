"""Tests for modulation waveforms."""

from typing import override
import numpy as np
from glyphtune import signal, waveforms


class TimePeriodicWave(waveforms.PeriodicWave):
    """Useless waveform that just returns the time signal when sampled."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return time


def test_phase_modulation_shifts_time_by_modulator() -> None:
    """Ensure that phase modulating a wave passes to it a time signal shifted by the modulator."""
    sampling_rate = 10
    duration = 1
    time_waveform = TimePeriodicWave(1)
    modulator = waveforms.Sine(1)

    phase_modulated_waveform = waveforms.phase_modulate(time_waveform, modulator)

    modulated_signal = phase_modulated_waveform.sample_seconds(sampling_rate, duration)
    time = time_waveform.sample_seconds(sampling_rate, duration)
    modulator_signal = modulator.sample_seconds(sampling_rate, duration)
    reference_signal = time + modulator_signal
    assert np.allclose(modulated_signal, reference_signal)
