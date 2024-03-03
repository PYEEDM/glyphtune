"""A hierarchy of classes for creating waveforms and sampling them into arrays of audio data."""

from glyphtune.waveforms.waveform import Waveform, OperationWaveform
from glyphtune.waveforms.periodic_waves import (
    PeriodicWave,
    Sine,
    Sawtooth,
    Pulse,
    Square,
    Triangle,
)
from glyphtune.waveforms.calculus import DerivativeWaveform, IntegralWaveform
from glyphtune.waveforms.modulation import (
    PhaseModulation,
    phase_modulate,
    amplitude_modulate,
    ring_modulate,
    frequency_modulate,
)

__all__ = [
    "Waveform",
    "OperationWaveform",
    "DerivativeWaveform",
    "IntegralWaveform",
    "PeriodicWave",
    "Sine",
    "Sawtooth",
    "Pulse",
    "Square",
    "Triangle",
    "PhaseModulation",
    "phase_modulate",
    "amplitude_modulate",
    "ring_modulate",
    "frequency_modulate",
]
