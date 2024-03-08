"""Waveforms resulting from modulating a carrier waveform by modulator waveforms."""

from typing import Any, override
import numpy as np
import glyphtune
from glyphtune.waveforms import periodic_waves, waveform


class PhaseModulation(waveform.Waveform):
    """Modulates a periodic wave's phase by another waveform's amplitude.

    Attributes:
        modulator: the modulator waveform.
    """

    def __init__(
        self, carrier: periodic_waves.PeriodicWave, modulator: waveform.Waveform
    ) -> None:
        """Initializes a phase modulation waveform.

        Args:
            carrier: the periodic wave whose phase will be modulated.
            modulator: the waveform used for the modulation.
        """
        self.carrier = carrier
        self.modulator = modulator

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

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        sampled_modulator = self.modulator.sample_arr(time_array)
        phase_modulation = sampled_modulator / self.carrier.frequency
        if self.__sinusoidal_carrier:
            phase_modulation /= 2 * np.pi
        modulated_time_array = time_array + phase_modulation
        return self.__carrier.sample_arr(modulated_time_array)

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PhaseModulation)
            and type(self) is type(other)
            and self.__carrier == other.carrier
            and self.modulator == other.modulator
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.__carrier}, {self.modulator})"


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
