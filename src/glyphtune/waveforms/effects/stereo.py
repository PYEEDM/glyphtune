"""Effects manipulating the stereo field of the input."""

from typing import Any, override
import numpy as np
import glyphtune
from glyphtune.waveforms import waveform
from glyphtune.waveforms.effects import effect


def is_stereo_signal(signal: glyphtune.FloatArray) -> bool:
    """Returns whether the given signal is stereo.

    Args:
        signal: input signal array.
    """
    return len(signal.shape) == 2 and signal.shape[0] == 2


class StereoPan(effect.Effect):
    """Stereo pan effect."""

    def __init__(
        self, input_waveform: waveform.Waveform, pan: float = 0, mix: float = 1
    ) -> None:
        """Initializes a stereo pan effect.

        Args:
            input_waveform: the input waveform of the effect.
            pan: stereo pan value in the range [-1, 1] i.e., from entirely left to entirely right.
            mix: the portion of the output that will be "wet". Can be negative for inverted output.
        """
        super().__init__(input_waveform, mix)
        self.__stereo_levels = StereoLevels(input_waveform)
        self.__stereo_inter_mix = StereoInterMix(self.__stereo_levels)
        self.pan = pan

    @property
    def pan(self) -> float:
        """Stereo pan value in the range [-1, 1] i.e., from entirely left to entirely right."""
        return self.__pan

    @pan.setter
    def pan(self, value: float) -> None:
        if abs(value) > 1:
            raise ValueError("Pan value must be in the range [-1, 1]")
        self.__pan = value
        self.__stereo_levels.left_level = 1 - max(value, 0)
        self.__stereo_levels.right_level = 1 + min(value, 0)
        self.__stereo_inter_mix.left_to_right = max(value, 0)
        self.__stereo_inter_mix.right_to_left = -min(value, 0)

    @override
    def apply(self, input_signal: glyphtune.FloatArray) -> glyphtune.FloatArray:
        if not is_stereo_signal(input_signal):
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        sampled_stereo_levels = self.__stereo_levels.apply(input_signal)
        sampled_stereo_inter_mix = self.__stereo_inter_mix.apply(input_signal)
        return sampled_stereo_levels + sampled_stereo_inter_mix

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, StereoPan)
            and self.__pan == other.pan
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        pan_repr = f", pan={self.__pan}" if self.__pan != 0 else ""
        return f"{class_name}({self.input_waveform}{pan_repr}{self._mix_repr(1)})"


class StereoLevels(effect.Effect):
    """An effect for setting each stereo channel's level independently.

    Attributes:
        left_level: the level of the left channel.
        right_level: the level of the right channel.
    """

    def __init__(
        self,
        input_waveform: waveform.Waveform,
        left_level: float = 1,
        right_level: float = 1,
        mix: float = 1,
    ) -> None:
        """Initializes a stereo levels effect.

        Args:
            input_waveform: the input waveform of the effect.
            left_level: the level of the left channel.
            right_level: the level of the right channel.
            mix: the portion of the output that will be "wet". Can be negative for inverted output.
        """
        super().__init__(input_waveform, mix)
        self.left_level = left_level
        self.right_level = right_level

    @override
    def apply(self, input_signal: glyphtune.FloatArray) -> glyphtune.FloatArray:
        if not is_stereo_signal(input_signal):
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        wet_signal: glyphtune.FloatArray
        wet_signal = input_signal * np.array([[self.left_level], [self.right_level]])
        return wet_signal

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, StereoLevels)
            and self.left_level == other.left_level
            and self.right_level == other.right_level
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        left_level_repr = (
            f", left_level={self.left_level}" if self.left_level != 1 else ""
        )
        right_level_repr = (
            f", right_level={self.right_level}" if self.right_level != 1 else ""
        )
        mix_repr = self._mix_repr(1)
        return f"{class_name}({self.input_waveform}{left_level_repr}{right_level_repr}{mix_repr})"


class StereoInterMix(effect.Effect):
    """An effect that sends signal from the left channel to the right channel and vice versa.

    Attributes:
        right_to_left: how much of the right channel to send to the left channel.
        left_to_right: how much of the left channel to send to the right channel.
    """

    def __init__(
        self,
        input_waveform: waveform.Waveform,
        right_to_left: float = 0,
        left_to_right: float = 0,
        mix: float = 1,
    ) -> None:
        """Initializes a stereo inter-mix effect.

        Args:
            input_waveform: the input waveform of the effect.
            right_to_left: how much of the right channel to send to the left channel.
            left_to_right: how much of the left channel to send to the right channel.
            mix: the portion of the output that will be "wet". Can be negative for inverted output.
        """
        super().__init__(input_waveform, mix)
        self.right_to_left = right_to_left
        self.left_to_right = left_to_right

    @override
    def apply(self, input_signal: glyphtune.FloatArray) -> glyphtune.FloatArray:
        if not is_stereo_signal(input_signal):
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        wet_signal: glyphtune.FloatArray = np.array(
            [
                self.right_to_left * input_signal[1],
                self.left_to_right * input_signal[0],
            ]
        )
        return wet_signal

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, StereoInterMix)
            and self.right_to_left == other.right_to_left
            and self.left_to_right == other.left_to_right
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        right_to_left_repr = (
            f", right_to_left={self.right_to_left}" if self.right_to_left != 0 else ""
        )
        left_to_right_repr = (
            f", left_to_right={self.left_to_right}" if self.left_to_right != 0 else ""
        )
        mix_repr = self._mix_repr(1)
        args_repr = (
            f"{self.input_waveform}{right_to_left_repr}{left_to_right_repr}{mix_repr}"
        )
        return f"{class_name}({args_repr})"


class StereoDelay(effect.Effect):
    """An effect that introduces a delay between the left and right channels (Haas effect).

    Attributes:
        left_right_delay: the delay between the left and right channels in seconds.
            A positive value means that the right channel's signal is delayed,
            and a negative value means that the left channel's signal is delayed.
    """

    def __init__(
        self,
        input_waveform: waveform.Waveform,
        left_right_delay: float = 0,
        mix: float = 1,
    ) -> None:
        """Initializes a stereo delay effect.

        Args:
            input_waveform: the input waveform of the effect.
            left_right_delay: the delay between the left and right channels in seconds.
                A positive value means that the right channel's signal is delayed,
                and a negative value means that the left channel's signal is delayed.
            mix: the portion of the output that will be "wet". Can be negative for inverted output.
        """
        super().__init__(input_waveform, mix)
        self.left_right_delay = left_right_delay

    @override
    def sample_dry_wet(
        self, time_array: glyphtune.FloatArray
    ) -> tuple[glyphtune.FloatArray, glyphtune.FloatArray]:
        if not is_stereo_signal(time_array):
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        dry_signal = self.input_waveform.sample_arr(time_array)
        wet_signal = dry_signal.copy()
        delay = abs(self.left_right_delay)
        if delay != 0:
            delayed_channel = 1 if self.left_right_delay > 0 else 0
            delayed_signal = self.input_waveform.sample_arr(time_array - delay)
            wet_signal[delayed_channel] = delayed_signal[delayed_channel]
        return dry_signal, wet_signal

    @override
    def apply(self, input_signal: glyphtune.FloatArray) -> glyphtune.FloatArray:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be applied directly and "
            "must be sampled using sample_dry_wet"
        )

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, StereoDelay)
            and self.left_right_delay == other.left_right_delay
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        left_rught_delay_repr = (
            f", left_right_delay={self.left_right_delay}"
            if self.left_right_delay != 0
            else ""
        )
        mix_repr = self._mix_repr(1)
        return f"{class_name}({self.input_waveform}{left_rught_delay_repr}{mix_repr})"
