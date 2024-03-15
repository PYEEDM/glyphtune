"""Waveforms produced by periodic functions."""

from typing import Any, override
import numpy as np
from glyphtune import arrays, _strings
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
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        return super().sample_arr(time_array)

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PeriodicWave)
            and type(self) is type(other)
            and self.frequency == other.frequency
            and self.phase == other.phase
        )

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.frequency}{self._phase_repr()})"

    def _to_local(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        """Returns local time array from a given time array.

        Equivalent to `self.frequency * time_array + self.phase`.

        Args:
            time_array: time array to convert.
        """
        return self.frequency * time_array + self.phase

    def _phase_repr(self, default_value: float = 0) -> str:
        return _strings.optional_param_repr("phase", default_value, self.phase)


class Sine(PeriodicWave):
    """Waveform with a sine shape."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        result: arrays.FloatArray = np.sin(2 * np.pi * self._to_local(time_array))
        return result


class Sawtooth(PeriodicWave):
    """Waveform with a sawtooth shape."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        local_time_array = self._to_local(time_array)
        result: arrays.FloatArray = 2 * (
            local_time_array - np.floor(local_time_array + 0.5)
        )
        return result


def _pulse_signal(
    time_array: arrays.FloatArray, duty_cycle: float
) -> arrays.FloatArray:
    result: arrays.FloatArray = (
        2 * (np.floor(time_array) - np.floor(time_array - duty_cycle)) - 1
    )
    return result


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
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        local_time_array = self._to_local(time_array)
        result = _pulse_signal(local_time_array, self.duty_cycle)
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Pulse)
            and super().__eq__(other)
            and self.duty_cycle == other.duty_cycle
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        phase_repr = self._phase_repr()
        duty_cycle_repr = _strings.optional_param_repr(
            "duty_cycle", 0.5, self.duty_cycle
        )
        return f"{class_name}({self.frequency}{phase_repr}{duty_cycle_repr})"


class Square(PeriodicWave):
    """Waveform with a square shape.

    Special case of a pulse wave where the duty cycle is equal to 0.5.
    """

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        local_time_array = self._to_local(time_array)
        result = _pulse_signal(local_time_array, 0.5)
        return result


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
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        local_time_array = self._to_local(time_array)
        offset_time_array = local_time_array + self.rising_part / 2
        sawtooth_signal = offset_time_array - np.floor(offset_time_array)
        result: arrays.FloatArray = np.piecewise(
            sawtooth_signal,
            [sawtooth_signal <= self.rising_part],
            [
                lambda x: -1 + 2 * (x / self.rising_part),
                lambda x: 1 - 2 * (x - self.rising_part) / (1 - self.rising_part),
            ],
        )
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Triangle)
            and super().__eq__(other)
            and self.rising_part == other.rising_part
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        phase_repr = self._phase_repr()
        rising_part_repr = _strings.optional_param_repr(
            "rising_part", 0.5, self.rising_part
        )
        return f"{class_name}({self.frequency}{phase_repr}{rising_part_repr})"
