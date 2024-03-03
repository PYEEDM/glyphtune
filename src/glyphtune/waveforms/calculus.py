"""Waveforms obtained by numerical differentiation or integration of other waveforms."""

from typing import Any, override
import numpy as np
import glyphtune
from glyphtune.waveforms import waveform


class DerivativeWaveform(waveform.Waveform):
    """A waveform that is the derivative of another waveform.

    Attributes:
        input_waveform: waveform being differentiated.
    """

    def __init__(self, input_waveform: waveform.Waveform) -> None:
        """Initializes a derivative waveform.

        Args:
            input_waveform: waveform to differentiate.
        """
        super().__init__()
        self.input_waveform = input_waveform

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        sampled_input_waveform = self.input_waveform.sample_arr(time_array)
        result: glyphtune.FloatArray = np.gradient(sampled_input_waveform, time_array)
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, DerivativeWaveform)
            and type(self) is type(other)
            and self.input_waveform == other.input_waveform
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.input_waveform})"


class IntegralWaveform(waveform.Waveform):
    """A waveform that is the integral of another waveform.

    Can use dynamic offset; When enabled, the integral waveform will store the last value of the
    integral computed by [`IntegralWaveform.sample_arr`][glyphtune.waveforms.Waveform.sample_arr]
    and offset the next result by it. This is useful for integrating waveforms in consecutive
    chunks without losing continuity.
    """

    def __init__(
        self, input_waveform: waveform.Waveform, dynamic_offset: bool = True
    ) -> None:
        """Initializes an integral waveform.

        Args:
            input_waveform: waveform to integrate.
            dynamic_offset: whether to use dynamic offset
                (see [`IntegralWaveform`][glyphtune.waveforms.IntegralWaveform] for info).
        """
        super().__init__()
        self.input_waveform = input_waveform
        self.dynamic_offset = dynamic_offset
        self.__offset = 0

    @property
    def input_waveform(self) -> waveform.Waveform:
        """Waveform being integrated."""
        return self.__input_waveform

    @input_waveform.setter
    def input_waveform(self, value: waveform.Waveform) -> None:
        self.__input_waveform = value
        self.reset_offset()

    @property
    def dynamic_offset(self) -> bool:
        """Whether this integral waveform uses dynamic offset
        (see [`IntegralWaveform`][glyphtune.waveforms.IntegralWaveform] for info)"""
        return self.__dynamic_offset

    @dynamic_offset.setter
    def dynamic_offset(self, value: bool) -> None:
        self.__dynamic_offset = value
        self.reset_offset()

    def reset_offset(self) -> None:
        """Resets the dynamic offset to the initial value of 0."""
        self.__offset = 0

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        sampled_input_waveform = self.input_waveform.sample_arr(time_array)
        scaled_sampled_input_waveform = sampled_input_waveform * np.gradient(time_array)
        result: glyphtune.FloatArray = np.cumsum(scaled_sampled_input_waveform)
        if self.__dynamic_offset:
            result += self.__offset
            self.__offset = result[-1]
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, IntegralWaveform)
            and type(self) is type(other)
            and self.__input_waveform == other.input_waveform
            and self.__dynamic_offset == other.dynamic_offset
        )

    @override
    def __repr__(self) -> str:
        class_name = type(self).__name__
        dynamic_offset_repr = (
            ", dynamic_offset=False" if not self.__dynamic_offset else ""
        )
        return f"{class_name}({self.input_waveform}{dynamic_offset_repr})"
