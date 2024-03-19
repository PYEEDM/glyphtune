"""Tests for operation waveforms."""

from typing import override
import numpy as np
from glyphtune import signal, waveforms


class DummyWaveform(waveforms.Waveform):
    """Useless waveform that always has a value of 1."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return signal.Signal(np.ones_like(time))


def test_adding_two_waveforms() -> None:
    """Ensure adding two waveforms together returns correct results."""
    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    addition_waveform = dummy_waveform + dummy_waveform

    addition_signal = addition_waveform.sample_seconds(duration, sampling_rate)
    reference_addition_signal = np.full((2, int(sampling_rate * duration)), 2)
    assert np.array_equal(addition_signal, reference_addition_signal)


def test_adding_waveform_and_float() -> None:
    """Ensure adding a waveform and a float together returns correct results."""
    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    addition_waveform = dummy_waveform + 2.3

    addition_signal = addition_waveform.sample_seconds(duration, sampling_rate)
    reference_addition_signal = np.full((2, int(sampling_rate * duration)), 3.3)
    assert np.array_equal(addition_signal, reference_addition_signal)


def test_multiplying_waveform_and_float() -> None:
    """Ensure multiplying a waveform and a float together returns correct results."""
    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    addition_waveform = dummy_waveform * 0.5

    addition_signal = addition_waveform.sample_seconds(duration, sampling_rate)
    reference_addition_signal = np.full((2, int(sampling_rate * duration)), 0.5)
    assert np.array_equal(addition_signal, reference_addition_signal)


def test_custom_operation_waveform() -> None:
    """Ensure operation waveforms with a custom operation and operands return correct results."""

    def add_mod(
        term1: signal.Signal,
        term2: signal.Signal,
        *,
        mod: signal.Signal | float = 1,
    ) -> signal.Signal:
        return signal.Signal((term1 + term2) % mod)

    sampling_rate = 10
    duration = 1
    dummy_waveform = DummyWaveform()

    add_mod_waveform = waveforms.OperationWaveform(add_mod, dummy_waveform, 4.2, mod=3)

    add_mod_signal = add_mod_waveform.sample_seconds(duration, sampling_rate)
    reference_add_mod_signal = np.full((2, int(sampling_rate * duration)), 2.2)
    assert np.array_equal(add_mod_signal, reference_add_mod_signal)
