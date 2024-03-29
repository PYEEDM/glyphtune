"""Basic functionality of effects."""

from typing import Any, final, override
from glyphtune import _strings, signal
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
            mix: multiplier of the wet signal. The dry signal will be multiplied by `1-abs(mix)`.
        """
        super().__init__()
        self.input_waveform = input_waveform
        self.mix = mix

    @property
    def mix(self) -> float:
        """Proportional multiplier of the effect's wet signal.

        The effect's wet signal will be multiplied by this value
        (therefore negative values mean inverted effect output).
        The dry signal will be multiplied by `1-abs(mix)`.
        """
        return self.__mix

    @mix.setter
    def mix(self, value: float) -> None:
        if abs(value) > 1:
            raise ValueError("Mix level must be in the range [-1, 1]")
        self.__mix = value

    @override
    @final
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        dry_signal, wet_signal = self.sample_dry_wet(time)
        dry_mix = 1 - abs(self.mix)
        return signal.Signal(dry_mix * dry_signal + self.mix * wet_signal)

    def sample_dry_wet(
        self, time: signal.Signal
    ) -> tuple[signal.Signal, signal.Signal]:
        """Samples the dry and wet signals of the effect given a time variable signal.

        Args:
            time: signal containing the values of the time variable at each sample point.

        Returns:
            A tuple `(dry_signal, wet_signal)`. Both `dry_signal` and `wet_signal` are of the same
                shape as `time`. `dry_signal` contains the signal of the input waveform
                without applying the effect, and `wet_signal` contains the signal of the applied
                effect.
        """
        dry_signal = self.input_waveform.sample_time(time)
        wet_signal = self.apply(dry_signal)
        return dry_signal, wet_signal

    def apply(self, input_signal: signal.Signal) -> signal.Signal:
        """Applies the effect on the given input signal.

        Effects that can generate output by simply altering the input audio should only have to
        implement this method. Effects that deal with more complex time-related manipulation should
        instead override `sample_dry_wet` and may just raise `NotImplementedError` in `apply`.

        Args:
            input_signal: input signal of the effect.

        Returns:
            The wet signal of the effect after being applied on `input_signal`.
                The returned signal will be of the same shape as `input_signal`.
        """
        raise NotImplementedError(
            f"Attempted to apply a base {type(self).__name__} object"
        )

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Effect)
            and type(self) is type(other)
            and self.input_waveform == other.input_waveform
            and self.mix == other.mix
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.input_waveform}{self._mix_repr()})"

    def _mix_repr(self, default_value: float = 0.5) -> str:
        return _strings.optional_param_repr("mix", default_value, self.mix)
