"""A hierarchy of classes for creating waveforms and sampling them into audio signals."""

from glyphtune.waveforms.waveform import Waveform, OperationWaveform
from glyphtune.waveforms.periodic_waves import (
    PeriodicWave,
    Sine,
    Sawtooth,
    Pulse,
    Square,
    Triangle,
)
from glyphtune.waveforms.modulation import (
    PhaseModulation,
    phase_modulate,
    amplitude_modulate,
    ring_modulate,
)

__all__ = [
    "Waveform",
    "OperationWaveform",
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
]
