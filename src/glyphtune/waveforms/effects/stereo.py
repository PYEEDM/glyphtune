"""Effects manipulating the stereo field of the input."""

from typing import override
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
