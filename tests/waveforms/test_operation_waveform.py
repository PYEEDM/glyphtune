"""Tests for operation waveforms."""

from typing import override
import numpy as np
from glyphtune import arrays, waveforms


class DummyWaveform(waveforms.Waveform):
    """Useless waveform that always has a value of 1."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        return np.ones_like(time_array)


def test_adding_two_waveforms() -> None:
    """Ensure adding two waveforms together returns correct results."""
    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    addition_waveform: waveforms.OperationWaveform = dummy_waveform + dummy_waveform

    sampled_addition_waveform = addition_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_addition_samples = np.full(int(sampling_rate * duration), 2)
    assert np.array_equal(sampled_addition_waveform, reference_addition_samples)


def test_adding_waveform_and_float() -> None:
    """Ensure adding a waveform and a float together returns correct results."""
    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    addition_waveform: waveforms.OperationWaveform = dummy_waveform + 2.3

    sampled_addition_waveform = addition_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_addition_samples = np.full(int(sampling_rate * duration), 3.3)
    assert np.array_equal(sampled_addition_waveform, reference_addition_samples)


def test_multiplying_waveform_and_float() -> None:
    """Ensure multiplying a waveform and a float together returns correct results."""
    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    addition_waveform: waveforms.OperationWaveform = dummy_waveform * 0.5

    sampled_addition_waveform = addition_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_addition_samples = np.full(int(sampling_rate * duration), 0.5)
    assert np.array_equal(sampled_addition_waveform, reference_addition_samples)


def test_custom_operation_waveform() -> None:
    """Ensure operation waveforms with a custom operation and operands return correct results."""

    def add_mod(
        term1: arrays.FloatArray,
        term2: arrays.FloatArray,
        *,
        mod: arrays.FloatArray | float = 1,
    ) -> arrays.FloatArray:
        return (term1 + term2) % mod

    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    add_mod_waveform = waveforms.OperationWaveform(add_mod, dummy_waveform, 4.2, mod=3)

    sampled_add_mod_waveform = add_mod_waveform.sample_seconds(sampling_rate, duration)
    reference_add_mod_samples = np.full(int(sampling_rate * duration), 2.2)
    assert np.array_equal(sampled_add_mod_waveform, reference_add_mod_samples)
