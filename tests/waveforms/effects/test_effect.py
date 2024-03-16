"""Tests for base effect class."""

from typing import override
import numpy as np
from glyphtune import signal, waveforms
from glyphtune.waveforms import effects


class TimeWaveform(waveforms.Waveform):
    """Useless waveform that just returns the time signal when sampled."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return time


class DummyEffect(effects.Effect):
    """Useless effect that just adds 1 to the input signal."""

    @override
    def apply(self, input_signal: signal.Signal) -> signal.Signal:
        return signal.Signal(input_signal + 1)


def test_sample_dry() -> None:
    """Ensure that sampling a completely dry effect just returns the original input."""
    sampling_rate = 10
    duration = 1
    time_waveform = TimeWaveform()

    dummy_effect = DummyEffect(time_waveform, 0)

    dummy_effect_signal = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_signal = time_waveform.sample_seconds(sampling_rate, duration)
    assert np.array_equal(dummy_effect_signal, reference_signal)


def test_sample_wet() -> None:
    """Ensure that sampling a completely wet effect returns the input with the effect applied."""
    sampling_rate = 10
    duration = 1
    time_waveform = TimeWaveform()

    dummy_effect = DummyEffect(time_waveform, 1)

    dummy_effect_signal = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_signal = time_waveform.sample_seconds(sampling_rate, duration) + 1
    assert np.array_equal(dummy_effect_signal, reference_signal)


def test_sample_negative_wet() -> None:
    """Ensure that sampling a completely wet effect returns the input with the effect applied."""
    sampling_rate = 10
    duration = 1
    time_waveform = TimeWaveform()

    dummy_effect = DummyEffect(time_waveform, -1)

    dummy_effect_signal = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_signal = -time_waveform.sample_seconds(sampling_rate, duration) - 1
    assert np.array_equal(dummy_effect_signal, reference_signal)


def test_sample_mixed() -> None:
    """Ensure that sampling a mixed effect returns the input with the effect partially applied."""
    sampling_rate = 10
    duration = 1
    time_waveform = TimeWaveform()

    dummy_effect = DummyEffect(time_waveform, 0.5)

    dummy_effect_signal = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_signal = time_waveform.sample_seconds(sampling_rate, duration) + 0.5
    assert np.allclose(dummy_effect_signal, reference_signal)


def test_sample_mixed_negative() -> None:
    """Ensure that sampling a mixed effect returns the input with the effect partially applied."""
    sampling_rate = 10
    duration = 1
    time_waveform = TimeWaveform()

    dummy_effect = DummyEffect(time_waveform, -0.5)

    dummy_effect_signal = dummy_effect.sample_seconds(sampling_rate, duration)
    assert np.allclose(dummy_effect_signal, -0.5)
