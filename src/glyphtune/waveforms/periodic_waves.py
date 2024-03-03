"""Waveforms produced by periodic functions."""

from typing import Any, override
import numpy as np
import glyphtune
from glyphtune.waveforms import waveform


class PeriodicWave(waveform.Waveform):
    """Periodic wave base class.

    Attributes:
        phase: initial phase offset of the periodic wave as a ratio of the period.
    """

    def __init__(self, frequency: float, phase: float = 0):
        """Initializes a periodic wave.

        Args:
            frequency: frequency of the periodic wave in Hz.
            phase: initial phase offset of the periodic wave as a ratio of the period.
        """
        super().__init__()
        self.frequency = frequency
        self.phase = phase

    @property
    def frequency(self) -> float:
        """Frequency of the periodic wave in Hz."""
        return self.__frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Frequency must be positive")
        self.__frequency = value

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        return super().sample_arr(time_array)

    def _to_local(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        """Returns local time array from a given time array.

        Equivalent to `self.frequency * time_array + self.phase`.

        Args:
            time_array: time array to convert.
        """
        return self.__frequency * time_array + self.phase

    def _phase_repr(self) -> str:
        return f", {self.phase}" if self.phase != 0 else ""


class Sine(PeriodicWave):
    """Waveform with a sine shape."""

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        result: glyphtune.FloatArray = np.sin(2 * np.pi * self._to_local(time_array))
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Sine)
            and type(self) is type(other)
            and self.__frequency == other.frequency
            and self.phase == other.phase
        )

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.frequency}{self._phase_repr()})"


class Sawtooth(PeriodicWave):
    """Waveform with a sawtooth shape."""

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        local_time_array = self._to_local(time_array)
        result: glyphtune.FloatArray = 2 * (
            local_time_array - np.floor(local_time_array + 0.5)
        )
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Sawtooth)
            and type(self) is type(other)
            and self.__frequency == other.frequency
            and self.phase == other.phase
        )

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.frequency}{self._phase_repr()})"


class Pulse(PeriodicWave):
    """Waveform with a pulse shape."""

    def __init__(
        self, frequency: float, phase: float = 0, duty_cycle: float = 0.5
    ) -> None:
        """Initializes a pulse wave.

        Args:
            frequency: frequency of the pulse wave in Hz.
            phase: initial phase offset as a ratio of the period.
            duty_cycle: the fraction of one period in which the signal is high.
        """
        super().__init__(frequency, phase)
        self.duty_cycle = duty_cycle

    @property
    def duty_cycle(self) -> float:
        """The fraction of one period in which the signal is high."""
        return self.__duty_cycle

    @duty_cycle.setter
    def duty_cycle(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError("duty_cycle must be in the range (0, 1)")
        self.__duty_cycle = value

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        local_time_array = self._to_local(time_array)
        result: glyphtune.FloatArray = (
            2
            * (
                np.floor(local_time_array)
                - np.floor(local_time_array - self.__duty_cycle)
            )
            - 1
        )
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Pulse)
            and type(self) is type(other)
            and self.__frequency == other.frequency
            and self.phase == other.phase
            and self.__duty_cycle == other.duty_cycle
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        phase_repr = self._phase_repr()
        duty_cycle_repr = (
            f", duty_cycle={self.duty_cycle}" if self.duty_cycle != 0.5 else ""
        )
        return f"{class_name}({self.frequency}{phase_repr}{duty_cycle_repr})"


class Square(Pulse):
    """Waveform with a square shape.

    Special case of a pulse wave where the duty cycle is equal to 0.5.
    """

    def __init__(self, frequency: float, phase: float = 0) -> None:
        """Initializes a square wave.

        Args:
            frequency: frequency of the square wave in Hz.
            phase: initial phase offset as a ratio of the period.
        """
        super().__init__(frequency, phase)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.frequency}{self._phase_repr()})"


class Triangle(PeriodicWave):
    """Waveform with a triangle shape."""

    def __init__(
        self, frequency: float, phase: float = 0, rising_part: float = 0.5
    ) -> None:
        """Initializes a triangle wave.

        Args:
            frequency: frequency of the square wave in Hz.
            phase: initial phase offset as a ratio of the period.
            rising_part: the fraction of one period in which the signal is rising.
        """
        super().__init__(frequency, phase)
        self.rising_part = rising_part

    @property
    def rising_part(self) -> float:
        """The fraction of one period in which the signal is rising."""
        return self.__rising_part

    @rising_part.setter
    def rising_part(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError("Rising part must be in the range (0, 1)")
        self.__rising_part = value

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        local_time_array = self._to_local(time_array)
        offset_time_array = local_time_array + self.__rising_part / 2
        sawtooth_signal = offset_time_array - np.floor(offset_time_array)
        result: glyphtune.FloatArray = np.piecewise(
            sawtooth_signal,
            [sawtooth_signal <= self.__rising_part],
            [
                lambda x: -1 + 2 * (x / self.__rising_part),
                lambda x: 1 - 2 * (x - self.__rising_part) / (1 - self.__rising_part),
            ],
        )
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Triangle)
            and type(self) is type(other)
            and self.__frequency == other.frequency
            and self.phase == other.phase
            and self.__rising_part == other.rising_part
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        phase_repr = self._phase_repr()
        rising_part_repr = (
            f", rising_part={self.rising_part}" if self.rising_part != 0.5 else ""
        )
        return f"{class_name}({self.frequency}{phase_repr}{rising_part_repr})"
