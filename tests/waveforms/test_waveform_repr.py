"""Tests for waveform representation."""

from typing import override
import numpy as np
import glyphtune
from glyphtune import waveforms


class DummyWaveform(waveforms.Waveform):
    """Useless waveform with a constant repr function."""

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        return super().sample_arr(time_array)

    @override
    def __repr__(self) -> str:
        return "DummyWaveform"


class DummyPeriodicWave(waveforms.PeriodicWave):
    """Useless waveform with a constant repr function."""

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        return super().sample_arr(time_array)

    @override
    def __repr__(self) -> str:
        return "DummyPeriodicWave"


def test_operation_waveform_repr_with_numpy_function_and_one_operand() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)

    negated_waveform: waveforms.OperationWaveform = -dummy_waveform

    reference_repr = f"numpy.negative({dummy_waveform_repr})"
    assert repr(negated_waveform) == reference_repr


def test_operation_waveform_repr_with_numpy_function_and_multiple_operands() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)

    addition_waveform: waveforms.OperationWaveform = dummy_waveform + dummy_waveform

    reference_repr = f"numpy.add({dummy_waveform_repr}, {dummy_waveform_repr})"
    assert repr(addition_waveform) == reference_repr


def test_operation_waveform_repr_with_no_operands() -> None:
    """Ensure the representation of the waveform is as expected."""
    nonsensical_waveform = waveforms.OperationWaveform(np.random.rand)

    reference_repr = "OperationWaveform(rand)"
    assert repr(nonsensical_waveform) == reference_repr


def test_operation_waveform_repr_with_no_operands_and_kwargs() -> None:
    """Ensure the representation of the waveform is as expected."""
    nonsensical_waveform = waveforms.OperationWaveform(
        np.random.rand, kwarg1=1, kwarg2=2
    )

    reference_repr = "OperationWaveform(rand, kwarg1=1, kwarg2=2)"
    assert repr(nonsensical_waveform) == reference_repr


def test_operation_waveform_repr_with_operands() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    nonsensical_waveform = waveforms.OperationWaveform(
        np.random.rand, dummy_waveform, dummy_waveform, dummy_waveform
    )

    reference_repr = (
        "OperationWaveform(rand, "
        f"{dummy_waveform_repr}, {dummy_waveform_repr}, {dummy_waveform_repr})"
    )
    assert repr(nonsensical_waveform) == reference_repr


def test_operation_waveform_repr_with_operands_and_kwargs() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    nonsensical_waveform = waveforms.OperationWaveform(
        np.add, dummy_waveform, dummy_waveform, kwarg=1
    )

    reference_repr = (
        "OperationWaveform(numpy.add, "
        f"{dummy_waveform_repr}, {dummy_waveform_repr}, kwarg=1)"
    )
    assert repr(nonsensical_waveform) == reference_repr


def test_sine_wave_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Sine(100)

    assert repr(wave) == "Sine(100)"


def test_sine_wave_repr_with_phase_argument() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Sine(100, phase=0.6)

    assert repr(wave) == "Sine(100, 0.6)"


def test_explicitly_specifying_default_phase_does_not_affect_repr() -> None:
    """Ensure explicitly providing the default phase argument does not affect the representation."""
    wave = waveforms.Sine(100, phase=0)

    assert repr(wave) == "Sine(100)"


def test_sawtooth_wave_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Sawtooth(100)

    assert repr(wave) == "Sawtooth(100)"


def test_pulse_wave_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Pulse(100)

    assert repr(wave) == "Pulse(100)"


def test_pulse_wave_repr_with_phase_argument() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Pulse(100, phase=0.6)

    assert repr(wave) == "Pulse(100, 0.6)"


def test_pulse_wave_repr_with_duty_cycle_argument() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Pulse(100, duty_cycle=0.6)

    assert repr(wave) == "Pulse(100, duty_cycle=0.6)"


def test_pulse_wave_repr_with_phase_and_duty_cycle_arguments() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Pulse(100, phase=0.6, duty_cycle=0.6)

    assert repr(wave) == "Pulse(100, 0.6, duty_cycle=0.6)"


def test_square_wave_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Square(100)

    assert repr(wave) == "Square(100)"


def test_triangle_wave_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Triangle(100)

    assert repr(wave) == "Triangle(100)"


def test_triangle_wave_repr_with_phase_argument() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Triangle(100, phase=0.6)

    assert repr(wave) == "Triangle(100, 0.6)"


def test_triangle_wave_repr_with_rising_part_argument() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Triangle(100, rising_part=0.6)

    assert repr(wave) == "Triangle(100, rising_part=0.6)"


def test_triangle_wave_repr_with_phase_and_rising_part_arguments() -> None:
    """Ensure the representation of the waveform is as expected."""
    wave = waveforms.Triangle(100, phase=0.6, rising_part=0.6)

    assert repr(wave) == "Triangle(100, 0.6, rising_part=0.6)"


def test_derivative_waveform_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)

    derivative_waveform = waveforms.DerivativeWaveform(dummy_waveform)

    assert repr(derivative_waveform) == f"DerivativeWaveform({dummy_waveform_repr})"


def test_integral_waveform_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)

    integral_waveform = waveforms.IntegralWaveform(dummy_waveform)

    assert repr(integral_waveform) == f"IntegralWaveform({dummy_waveform_repr})"


def test_integral_waveform_repr_without_dynamic_offset() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)

    integral_waveform = waveforms.IntegralWaveform(dummy_waveform, dynamic_offset=False)

    assert (
        repr(integral_waveform)
        == f"IntegralWaveform({dummy_waveform_repr}, dynamic_offset=False)"
    )


def test_phase_modulation_waveform_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_periodic_wave = DummyPeriodicWave(1)
    dummy_wave_repr = repr(dummy_periodic_wave)

    phase_modulation_waveform = waveforms.phase_modulate(
        dummy_periodic_wave, dummy_periodic_wave
    )

    assert (
        repr(phase_modulation_waveform)
        == f"PhaseModulation({dummy_wave_repr}, {dummy_wave_repr})"
    )


def test_frequency_modulation_waveform_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_periodic_wave = DummyPeriodicWave(1)
    dummy_wave_repr = repr(dummy_periodic_wave)

    phase_modulation_waveform = waveforms.frequency_modulate(
        dummy_periodic_wave, dummy_periodic_wave
    )

    assert (
        repr(phase_modulation_waveform)
        == f"PhaseModulation({dummy_wave_repr}, {dummy_wave_repr}, frequency_modulation=True)"
    )
