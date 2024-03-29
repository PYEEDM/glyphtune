"""Effects manipulating the stereo field of the input."""

from typing import Any, override
import numpy as np
from glyphtune import _strings, signal
from glyphtune.waveforms import waveform
from glyphtune.waveforms.effects import effect


class StereoPan(effect.Effect):
    """Stereo pan effect."""

    def __init__(
        self, input_waveform: waveform.Waveform, pan: float = 0, mix: float = 1
    ) -> None:
        """Initializes a stereo pan effect.

        Args:
            input_waveform: the input waveform of the effect.
            pan: stereo pan value in the range [-1, 1] i.e., from entirely left to entirely right.
            mix: multiplier of the wet signal. The dry signal will be multiplied by `1-abs(mix)`.
        """
        super().__init__(input_waveform, mix)
        self._stereo_levels = StereoLevels(input_waveform)
        self._stereo_inter_mix = StereoInterMix(self._stereo_levels, mix=1)
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
        self._stereo_levels.left_level = 1 - max(value, 0)
        self._stereo_levels.right_level = 1 + min(value, 0)
        self._stereo_inter_mix.left_to_right = max(value, 0)
        self._stereo_inter_mix.right_to_left = -min(value, 0)

    @override
    def apply(self, input_signal: signal.Signal) -> signal.Signal:
        if not input_signal.is_stereo:
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        stereo_levels_signal = self._stereo_levels.apply(input_signal)
        stereo_inter_mix_signal = self._stereo_inter_mix.apply(input_signal)
        return signal.Signal(stereo_levels_signal + stereo_inter_mix_signal)

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, StereoPan)
            and self.pan == other.pan
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        pan_repr = _strings.optional_param_repr("pan", 0, self.pan)
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
            mix: multiplier of the wet signal. The dry signal will be multiplied by `1-abs(mix)`.
        """
        super().__init__(input_waveform, mix)
        self.left_level = left_level
        self.right_level = right_level

    @override
    def apply(self, input_signal: signal.Signal) -> signal.Signal:
        if not input_signal.is_stereo:
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        wet_signal = signal.Signal(
            input_signal * [[self.left_level], [self.right_level]]
        )
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
        left_level_repr = _strings.optional_param_repr("left_level", 1, self.left_level)
        right_level_repr = _strings.optional_param_repr(
            "right_level", 1, self.right_level
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
        right_to_left: float = 1,
        left_to_right: float = 1,
        mix: float = 0.5,
    ) -> None:
        """Initializes a stereo inter-mix effect.

        Args:
            input_waveform: the input waveform of the effect.
            right_to_left: how much of the right channel to send to the left channel.
            left_to_right: how much of the left channel to send to the right channel.
            mix: multiplier of the wet signal. The dry signal will be multiplied by `1-abs(mix)`.
        """
        super().__init__(input_waveform, mix)
        self.right_to_left = right_to_left
        self.left_to_right = left_to_right

    @override
    def apply(self, input_signal: signal.Signal) -> signal.Signal:
        if not input_signal.is_stereo:
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        wet_signal = signal.Signal(
            np.array(
                [
                    self.right_to_left * input_signal[1],
                    self.left_to_right * input_signal[0],
                ]
            )
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
        right_to_left_repr = _strings.optional_param_repr(
            "right_to_left", 1, self.right_to_left
        )
        left_to_right_repr = _strings.optional_param_repr(
            "left_to_right", 1, self.left_to_right
        )
        mix_repr = self._mix_repr(0.5)
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
            mix: multiplier of the wet signal. The dry signal will be multiplied by `1-abs(mix)`.
        """
        super().__init__(input_waveform, mix)
        self.left_right_delay = left_right_delay

    @override
    def sample_dry_wet(
        self, time: signal.Signal
    ) -> tuple[signal.Signal, signal.Signal]:
        if not time.is_stereo:
            raise ValueError(
                f"{type(self).__name__} can only be applied to a stereo signal"
            )
        dry_signal = self.input_waveform.sample_time(time)
        wet_signal = signal.Signal(np.copy(dry_signal))
        delay = abs(self.left_right_delay)
        if delay != 0:
            delayed_channel = 1 if self.left_right_delay > 0 else 0
            delayed_signal = self.input_waveform.sample_time(time - delay)
            wet_signal[delayed_channel] = delayed_signal[delayed_channel]
        return dry_signal, wet_signal

    @override
    def apply(self, input_signal: signal.Signal) -> signal.Signal:
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
        left_right_delay_repr = _strings.optional_param_repr(
            "left_right_delay", 0, self.left_right_delay
        )
        mix_repr = self._mix_repr(1)
        return f"{class_name}({self.input_waveform}{left_right_delay_repr}{mix_repr})"
