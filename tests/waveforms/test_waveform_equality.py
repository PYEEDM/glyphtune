"""Tests for waveform equality checks."""

from typing import Any, override
import numpy as np
from glyphtune import signal, waveforms


class DummyWaveform(waveforms.Waveform):
    """Useless waveform that's always equal to any other waveform of its kind."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return super().sample_time(time)

    @override
    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other)


def test_operation_waveform_equality() -> None:
    """Ensure equality check result is as expected."""
    addition = DummyWaveform() + DummyWaveform()
    also_addition = waveforms.OperationWaveform(
        np.add, DummyWaveform(), DummyWaveform()
    )
    assert addition == also_addition


def test_operation_waveform_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    addition = DummyWaveform() + DummyWaveform()
    assert addition != DummyWaveform()


def test_operation_waveform_inequality_operator() -> None:
    """Ensure equality check result is as expected."""
    addition = DummyWaveform() + DummyWaveform()
    not_addition = waveforms.OperationWaveform(
        np.subtract, DummyWaveform(), DummyWaveform()
    )
    assert addition != not_addition


def test_operation_waveform_inequality_operands() -> None:
    """Ensure equality check result is as expected."""
    addition = DummyWaveform() + DummyWaveform()
    too_much_addition = waveforms.OperationWaveform(
        np.add, DummyWaveform(), DummyWaveform(), DummyWaveform()
    )
    assert addition != too_much_addition


def test_operation_waveform_inequality_kwargs() -> None:
    """Ensure equality check result is as expected."""
    addition = DummyWaveform() + DummyWaveform()
    also_addition = waveforms.OperationWaveform(
        np.add, DummyWaveform(), DummyWaveform(), kwarg="test"
    )
    assert addition != also_addition


def test_periodic_wave_equality() -> None:
    """Ensure equality check result is as expected."""
    periodic_wave = waveforms.PeriodicWave(100, 1)
    also_periodic_wave = waveforms.PeriodicWave(100, 1)
    assert periodic_wave == also_periodic_wave


def test_periodic_wave_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    periodic_wave = waveforms.PeriodicWave(100)
    not_periodic_wave = DummyWaveform()
    assert periodic_wave != not_periodic_wave


def test_periodic_wave_inequality_frequency() -> None:
    """Ensure equality check result is as expected."""
    periodic_wave = waveforms.PeriodicWave(100)
    other_periodic_wave = waveforms.PeriodicWave(101)
    assert periodic_wave != other_periodic_wave


def test_periodic_wave_inequality_phase() -> None:
    """Ensure equality check result is as expected."""
    periodic_wave = waveforms.PeriodicWave(100)
    other_periodic_wave = waveforms.PeriodicWave(100, 0.1)
    assert periodic_wave != other_periodic_wave


def test_sine_equality() -> None:
    """Ensure equality check result is as expected."""
    sine = waveforms.Sine(1)
    also_sine = waveforms.Sine(1)
    assert sine == also_sine


def test_sine_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    sine = waveforms.Sine(1)
    not_sine = waveforms.PeriodicWave(1)
    assert sine != not_sine


def test_pulse_equality() -> None:
    """Ensure equality check result is as expected."""
    pulse = waveforms.Pulse(1, duty_cycle=0.4)
    also_pulse = waveforms.Pulse(1, duty_cycle=0.4)
    assert pulse == also_pulse


def test_pulse_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    pulse = waveforms.Pulse(1)
    not_pulse = waveforms.PeriodicWave(1)
    assert pulse != not_pulse


def test_pulse_inequality_duty_cycle() -> None:
    """Ensure equality check result is as expected."""
    pulse = waveforms.Pulse(1)
    other_pulse = waveforms.Pulse(1, duty_cycle=0.2)
    assert pulse != other_pulse


def test_triangle_equality() -> None:
    """Ensure equality check result is as expected."""
    triangle = waveforms.Triangle(1, rising_part=0.4)
    also_triangle = waveforms.Triangle(1, rising_part=0.4)
    assert triangle == also_triangle


def test_triangle_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    triangle = waveforms.Triangle(1)
    not_triangle = waveforms.PeriodicWave(1)
    assert triangle != not_triangle


def test_triangle_inequality_duty_cycle() -> None:
    """Ensure equality check result is as expected."""
    triangle = waveforms.Triangle(1)
    other_triangle = waveforms.Triangle(1, rising_part=0.2)
    assert triangle != other_triangle


def test_phase_modulation_equality() -> None:
    """Ensure equality check result is as expected."""
    phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1), waveforms.Sine(2, 0.1)
    )
    also_phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1), waveforms.Sine(2, 0.1)
    )
    assert phase_modulation == also_phase_modulation


def test_phase_modulation_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1), waveforms.Sine(2, 0.1)
    )
    not_phase_modulation = waveforms.ring_modulate(
        waveforms.Sine(1), waveforms.Sine(2, 0.1)
    )
    assert phase_modulation != not_phase_modulation


def test_phase_modulation_inequality_carrier() -> None:
    """Ensure equality check result is as expected."""
    phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1), waveforms.Sine(2, 0.1)
    )
    other_phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1, 0.1), waveforms.Sine(2, 0.1)
    )
    assert phase_modulation != other_phase_modulation


def test_phase_modulation_inequality_modulator() -> None:
    """Ensure equality check result is as expected."""
    phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1), waveforms.Sine(2, 0.1)
    )
    other_phase_modulation = waveforms.phase_modulate(
        waveforms.Sine(1), waveforms.Sine(2)
    )
    assert phase_modulation != other_phase_modulation
