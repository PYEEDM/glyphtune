"""Basic functionality of effects."""

from typing import final, override
import glyphtune
from glyphtune.waveforms import waveform


class Effect(waveform.Waveform):
    """Base class representing effects.

    Attributes:
        input_waveform: the input waveform of the effect.
    """

    def __init__(self, input_waveform: waveform.Waveform, mix: float = 0.5) -> None:
        """Initializes an effect.

        Args:
            input_waveform: the input waveform of the effect.
            mix: the portion of the output that will be "wet". Can be negative for inverted output.
        """
        self.input_waveform = input_waveform
        self.mix = mix

    @property
    def mix(self) -> float:
        """The portion of the output that will be "wet" Can be negative for inverted output."""
        return self.__mix

    @mix.setter
    def mix(self, value: float) -> None:
        if abs(value) > 1:
            raise ValueError("Mix level must be in the range [-1, 1]")
        self.__mix = value

    @override
    @final
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        dry_signal, wet_signal = self.sample_dry_wet(time_array)
        dry_mix = 1 - abs(self.mix)
        return dry_mix * dry_signal + self.__mix * wet_signal

    def sample_dry_wet(
        self, time_array: glyphtune.FloatArray
    ) -> tuple[glyphtune.FloatArray, glyphtune.FloatArray]:
        """Samples the dry and wet signals of the effect given a time variable array.

        Args:
            time_array: a 2d array of the shape `(channels, samples)` containing, for each channel,
                the values of the time variable at each sample point.

        Returns:
            A tuple `(dry_signal, wet_signal)`. Both `dry_signal` and `wet_signal` are of the same
                shape as `time_array`. `dry_signal` contains the signal of the input waveform
                without applying the effect, and `wet_signal` contains the signal of the applied
                effect.
        """
        dry_signal = self.input_waveform.sample_arr(time_array)
        wet_signal = self.apply(dry_signal)
        return dry_signal, wet_signal

    def apply(self, input_signal: glyphtune.FloatArray) -> glyphtune.FloatArray:
        """Applies the effect on the given input signal.

        Effects that can generate output by simply altering the input audio should only have to
        implement this method. Effects that deal with more complex time-related manipulation should
        instead override `sample_dry_wet` and may just raise `NotImplementedError` in `apply`.

        Args:
            input_signal: 2d array of the shape `(channels, samples)` containing the input signal.

        Returns:
            An array containing the wet signal of the effect after being applied on `input_signal`.
                The returned array will be of the same shape as `input_signal`.
        """
        raise NotImplementedError(
            f"Attempted to apply a base {type(self).__name__} object"
        )
