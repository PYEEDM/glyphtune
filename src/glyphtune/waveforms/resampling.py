"""Waveforms for resampling audio data."""

from __future__ import annotations
from typing import Any, override
import sys
import numpy as np
from glyphtune import _strings, signal
from glyphtune.waveforms import waveform


class ResampleWaveform(waveform.Waveform):
    """Waveform for resampling audio data.

    Attributes:
        time_multiplier: time/speed multiplier. Can be negative for reverse resampling.
        loop: whether to replay the sample after it finishes.
    """

    def __init__(
        self,
        original_audio: signal.Signal,
        sampling_rate: int,
        time_multiplier: float = 1,
        loop: bool = False,
    ) -> None:
        """Initializes a resample waveform with audio data.

        Args:
            original_audio: signal containing the audio data to resample.
            sampling_rate: the original sampling rate of `original_audio`, in samples per second.
            time_multiplier: time/speed multiplier. Can be negative for reverse resampling.
            loop: whether to replay the sample after it finishes.
        """
        super().__init__()
        self.original_audio = original_audio
        self.sampling_rate = sampling_rate
        self.time_multiplier = time_multiplier
        self.loop = loop

    @property
    def original_audio(self) -> signal.Signal:
        """Signal containing the audio data to resample."""
        return self.__original_audio

    @original_audio.setter
    def original_audio(self, value: signal.Signal) -> None:
        self.__original_audio = signal.Signal(value)

    @property
    def sampling_rate(self) -> int:
        """The original sampling rate of the audio data, in samples per second."""
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Sampling rate must be positive")
        self.__sampling_rate = value

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        if self.original_audio.channels == time.channels:
            audio = self.original_audio
        elif self.original_audio.is_mono and not time.is_mono:
            audio = self.original_audio.expand_channels(time.channels)
        elif time.is_mono and not self.original_audio.is_mono:
            audio = self.original_audio.to_mono()
        else:
            raise ValueError(
                "Time signal and audio data have incompatible channel counts"
            )
        scaled_time = time * self.sampling_rate * self.time_multiplier
        sample_indices = np.array(np.round(scaled_time), dtype=int)
        if self.time_multiplier < 0:
            sample_indices += audio.length - 1
        if self.loop:
            sample_indices %= audio.length
            indices_outside_data = np.empty(0, dtype=int)
        else:
            indices_outside_data = np.logical_or(
                sample_indices < 0, sample_indices >= audio.length
            )
            sample_indices[indices_outside_data] = 0
        result = signal.Signal(np.take_along_axis(audio.data, sample_indices, axis=1))
        result[indices_outside_data] = 0
        return result

    def full_repr(self) -> str:
        """Returns the string representation including the audio data array in its entirety."""
        with np.printoptions(threshold=sys.maxsize):
            return repr(self)

    def approx_equal(
        self,
        other: Any,
        absolute_tolerance: float = 1.0e-5,
        relative_tolerance: float = 1.0e-8,
    ) -> bool:
        """Returns whether this waveform is approximately equal to another within a tolerence.

        Args:
            other: waveform to compare to.
            absolute_tolerance: the absolute tolerance of the comparison.
            relative_tolerance: the relative tolerance of the comparison.
        """
        return (
            isinstance(other, ResampleWaveform)
            and type(self) is type(other)
            and np.allclose(
                self.original_audio,
                other.original_audio,
                relative_tolerance,
                absolute_tolerance,
            )
            and self.sampling_rate == other.sampling_rate
            and self.time_multiplier == other.time_multiplier
            and self.loop == other.loop
        )

    @override
    def __eq__(self, other: Any) -> Any:
        return (
            isinstance(other, ResampleWaveform)
            and type(self) is type(other)
            and np.array_equal(self.original_audio, other.original_audio)
            and self.sampling_rate == other.sampling_rate
            and self.time_multiplier == other.time_multiplier
            and self.loop == other.loop
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        time_multiplier_repr = _strings.optional_param_repr(
            "time_multiplier", 1, self.time_multiplier
        )
        loop_repr = _strings.optional_param_repr("loop", False, self.loop)
        return (
            f"{class_name}("
            f"{self.original_audio}, "
            f"{self.sampling_rate}"
            f"{time_multiplier_repr}"
            f"{loop_repr}"
            ")"
        )
