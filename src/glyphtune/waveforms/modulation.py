"""Waveforms resulting from modulating a carrier waveform by modulator waveforms."""

from typing import Any, override
import numpy as np
import glyphtune
from glyphtune.waveforms import calculus, periodic_waves, waveform


class PhaseModulation(waveform.Waveform):
    """Modulates a periodic wave's phase/frequency by another waveform's amplitude.

    Attributes:
        frequency_modulation: whether the frequency of the carrier will be modulated
            rather than the phase.
    """

    def __init__(
        self,
        carrier: periodic_waves.PeriodicWave,
        modulator: waveform.Waveform,
        frequency_modulation: bool = False,
    ) -> None:
        """Initializes a phase modulation waveform.

        Args:
            carrier: the periodic wave whose phase/frequency will be modulated.
            modulator: the waveform used for the modulation.
            frequency_modulation: whether the frequency of the carrier will be modulated rather
                than the phase.
        """
        self.carrier = carrier
        self.modulator = modulator
        self.frequency_modulation = frequency_modulation

    @property
    def carrier(self) -> periodic_waves.PeriodicWave:
        """The periodic carrier of the modulation."""
        return self.__carrier

    @carrier.setter
    def carrier(self, value: periodic_waves.PeriodicWave) -> None:
        if not isinstance(value, periodic_waves.PeriodicWave):
            raise TypeError("Phase modulation carrier must be a periodic wave")
        self.__carrier = value
        self.__sinusoidal_carrier = isinstance(self.carrier, periodic_waves.Sine)

    @property
    def modulator(self) -> waveform.Waveform:
        """The modulator waveform."""
        return self.__modulator

    @modulator.setter
    def modulator(self, value: waveform.Waveform) -> None:
        self.__modulator = value
        self.__modulator_integral = calculus.IntegralWaveform(self.__modulator)

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        sampled_modulator = (
            self.__modulator_integral.sample_arr(time_array)
            if self.frequency_modulation
            else self.__modulator.sample_arr(time_array)
        )
        phase_modulation = sampled_modulator / self.carrier.frequency
        if self.__sinusoidal_carrier and not self.frequency_modulation:
            phase_modulation /= 2 * np.pi
        modulated_time_array = time_array + phase_modulation
        return self.__carrier.sample_arr(modulated_time_array)

    @override
    def __eq__(self, other: Any) -> bool:
        if not type(self) is type(other):
            return False
        assert isinstance(other, PhaseModulation)
        return (
            self.__carrier == other.carrier
            and self.__modulator == other.modulator
            and self.frequency_modulation == other.frequency_modulation
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        frequency_modulation_repr = (
            ", frequency_modulation=True" if self.frequency_modulation else ""
        )
        return f"{class_name}({self.__carrier}, {self.__modulator}{frequency_modulation_repr})"


def phase_modulate(
    carrier: periodic_waves.PeriodicWave, *modulators: waveform.Waveform
) -> waveform.Waveform:
    """Returns a waveform resulting from phase modulation.

    The carrier's phase is shifted according to the amplitude of the sum of the modulators.

    Args:
        carrier: the periodic carrier waveform.
        *modulators: any number of modulator waveforms.
    """
    if not modulators:
        return carrier
    modulator_sum: waveform.Waveform = np.sum(list(modulators))
    return PhaseModulation(carrier, modulator_sum)


def amplitude_modulate(
    carrier: waveform.Waveform, *modulators: waveform.Waveform
) -> waveform.Waveform:
    """Returns a waveform resulting from amplitude modulation.

    The carrier's amplitude is shifted according to the amplitude of the sum of the modulators.

    Args:
        carrier: the carrier waveform.
        *modulators: any number of modulator waveforms.
    """
    if not modulators:
        return carrier
    result: waveform.Waveform = carrier + np.sum(list(modulators))
    return result


def ring_modulate(
    carrier: waveform.Waveform, *modulators: waveform.Waveform
) -> waveform.Waveform:
    """Returns a waveform resulting from ring modulation.

    The carrier's amplitude is multiplied by the amplitude of the product of the modulators.

    Args:
        carrier: the carrier waveform.
        *modulators: any number of modulator waveforms.
    """
    if not modulators:
        return carrier
    # mypy refuses to understand that this is fine, even though it works perfectly for np.sum...
    result: waveform.Waveform = carrier * np.prod(list(modulators))  # type: ignore[arg-type]
    return result


def frequency_modulate(
    carrier: periodic_waves.PeriodicWave, *modulators: waveform.Waveform
) -> waveform.Waveform:
    """Returns a waveform resulting from frequency modulation.

    The carrier's phase is shifted according to the amplitude of the integral of the sum of the
    modulators.

    Args:
        carrier: the periodic carrier waveform.
        *modulators: any number of modulator waveforms.
    """
    if not modulators:
        return carrier
    modulator_sum: waveform.Waveform = np.sum(list(modulators))
    return PhaseModulation(carrier, modulator_sum, frequency_modulation=True)
