"""Waveforms resulting from modulating a carrier waveform by modulator waveforms."""

from typing import Any, override
import numpy as np
from glyphtune import signal
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
        super().__init__()
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

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        modulator_signal = self.modulator.sample_time(time)
        phase_modulation = modulator_signal / self.carrier.frequency
        if isinstance(self.carrier, periodic_waves.Sine):
            phase_modulation /= 2 * np.pi
        modulated_time = time + phase_modulation
        return self.carrier.sample_time(modulated_time)

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PhaseModulation)
            and type(self) is type(other)
            and self.carrier == other.carrier
            and self.modulator == other.modulator
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.carrier}, {self.modulator})"


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
    modulator_sum = np.sum(list(modulators))
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
    result = carrier + np.sum(list(modulators))
    assert isinstance(result, waveform.Waveform)
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
    result = carrier * np.prod(list(modulators))  # type: ignore[arg-type]
    assert isinstance(result, waveform.Waveform)
    return result
