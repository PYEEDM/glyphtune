"""Tests for base effect class."""

from typing import override
import numpy as np
import pytest
from glyphtune import arrays, waveforms
from glyphtune.waveforms import effects


class TimeArrayWaveform(waveforms.Waveform):
    """Useless waveform that just returns the time array when sampled."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        return time_array


class DummyEffect(effects.Effect):
    """Useless effect that just adds 1 to the input signal."""

    @override
    def apply(self, input_signal: arrays.FloatArray) -> arrays.FloatArray:
        return input_signal + 1


def test_sample_dry() -> None:
    """Ensure that sampling a completely dry effect just returns the original input."""
    sampling_rate = 10
    duration = 1
    time_array_waveform = TimeArrayWaveform()

    dummy_effect = DummyEffect(time_array_waveform, 0)

    sampled_dummy_effect = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_samples = time_array_waveform.sample_seconds(sampling_rate, duration)
    assert np.array_equal(sampled_dummy_effect, reference_samples)


def test_sample_wet() -> None:
    """Ensure that sampling a completely wet effect returns the input with the effect applied."""
    sampling_rate = 10
    duration = 1
    time_array_waveform = TimeArrayWaveform()

    dummy_effect = DummyEffect(time_array_waveform, 1)

    sampled_dummy_effect = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_samples = time_array_waveform.sample_seconds(sampling_rate, duration) + 1
    assert np.array_equal(sampled_dummy_effect, reference_samples)


def test_sample_negative_wet() -> None:
    """Ensure that sampling a completely wet effect returns the input with the effect applied."""
    sampling_rate = 10
    duration = 1
    time_array_waveform = TimeArrayWaveform()

    dummy_effect = DummyEffect(time_array_waveform, -1)

    sampled_dummy_effect = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_samples = -time_array_waveform.sample_seconds(sampling_rate, duration) - 1
    assert np.array_equal(sampled_dummy_effect, reference_samples)


def test_sample_mixed() -> None:
    """Ensure that sampling a mixed effect returns the input with the effect partially applied."""
    sampling_rate = 10
    duration = 1
    time_array_waveform = TimeArrayWaveform()

    dummy_effect = DummyEffect(time_array_waveform, 0.5)

    sampled_dummy_effect = dummy_effect.sample_seconds(sampling_rate, duration)
    reference_samples = (
        time_array_waveform.sample_seconds(sampling_rate, duration) + 0.5
    )
    assert sampled_dummy_effect == pytest.approx(reference_samples)


def test_sample_mixed_negative() -> None:
    """Ensure that sampling a mixed effect returns the input with the effect partially applied."""
    sampling_rate = 10
    duration = 1
    time_array_waveform = TimeArrayWaveform()

    dummy_effect = DummyEffect(time_array_waveform, -0.5)

    sampled_dummy_effect = dummy_effect.sample_seconds(sampling_rate, duration)
    assert np.allclose(sampled_dummy_effect, -0.5)
