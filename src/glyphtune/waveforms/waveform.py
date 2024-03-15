"""Basic functionality of waveforms and operations on them."""

from __future__ import annotations
from typing import Any, Callable, Literal, override
import numpy as np
from glyphtune import arrays, _strings


class Waveform(np.lib.mixins.NDArrayOperatorsMixin):
    """Base class representing waveforms.

    Waveform objects do not store any audio data. Instead, they must generate that data when
    one of their sampling functions is called.

    Waveform objects can be used as operands with Python operators and as inputs to NumPy functions
    along with other waveforms and numbers. This returns a
    [`waveforms.OperationWaveform`][glyphtune.waveforms.OperationWaveform].
    """

    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        """Samples audio data given a time variable array.

        Args:
            time_array: a 2d array of the shape `(channels, samples)` containing, for each channel,
                the values of the time variable at each sample point.

        Returns:
            An array containing the sampled signal of the waveform.
                The returned array will be of the same shape as `time_array`.
                Each value in the returned array is the sample of the waveform's signal at the
                corresponding time value (in seconds) for a particular channel in `time_array`.
        """
        raise NotImplementedError(
            f"Attempted to sample a base {type(self).__name__} object"
        )

    def sample_seconds(
        self,
        sampling_rate: int,
        duration: float,
        start_offset: float = 0,
        channels: int = 2,
    ) -> arrays.FloatArray:
        """Samples audio data given time information in seconds.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            duration: the duration of time to be sampled in seconds.
            start_offset: the starting offset in seconds.
            channels: the number of channels to return.

        Returns:
            A 2d array containing the sampled signal of the waveform.
                The returned array will have the shape `(channels, ceil(sampling_rate*duration))`,
                containing `ceil(sampling_rate*duration)` samples for each channel.
        """
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if channels <= 0:
            raise ValueError("Number of channels must be positive")
        end = start_offset + duration
        sampling_period = 1 / sampling_rate
        time_array = np.arange(start_offset, end, sampling_period)
        time_array = np.tile(time_array, (channels, 1))
        return self.sample_arr(time_array)

    def sample_samples(
        self, sampling_rate: int, count: int, start_offset: int = 0, channels: int = 2
    ) -> arrays.FloatArray:
        """Samples audio data given sample count information.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            count: the number of samples to take.
            start_offset: the starting offset in samples.
            channels: the number of channels to return.

        Returns:
            A 2d array containing the sampled signal of the waveform.
                The returned array will have the shape `(channels, count)`,
                containing `count` samples for each channel.
        """
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if channels <= 0:
            raise ValueError("Number of channels must be positive")
        duration_seconds = count / sampling_rate
        start_offset_seconds = start_offset / sampling_rate
        return self.sample_seconds(
            sampling_rate, duration_seconds, start_offset_seconds, channels
        )

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
        ],
        *inputs: Any,
        **kwargs: Any,
    ) -> OperationWaveform:
        if method != "__call__" or "out" in kwargs:
            return NotImplemented
        for operand in inputs:
            if not isinstance(operand, (Waveform, float, int)):
                return NotImplemented
        return OperationWaveform(ufunc, *inputs, **kwargs)


class OperationWaveform(Waveform):
    """Waveform that is the result of an operation on other waveforms and numbers.

    "lazily" samples audio by recursively sampling the operands as needed and applying the
    operator on the resulting audio data. The operands may be waveforms or scalar values.

    Attributes:
        operator: the operator to be called on the (sampled) operands when this waveform is sampled.
        operands: the operands whose (sampled) outputs are passed to the operator when this
            waveform is sampled.
        operator_kwargs: the keyword arguments this waveform passes to its operator when sampled.
    """

    def __init__(
        self,
        operator: Callable[..., Any],
        *operands: Waveform | float,
        **operator_kwargs: Any,
    ):
        """Initializes an operation waveform with operands and operator information.

        Args:
            operator: the operator to use on the (sampled) operands when the waveform is sampled.
                The operator should take float arrays/numbers as input and return a float array.
            *operands: operands whose (sampled) outputs are passed to the operator when the waveform
                is sampled.
            **operator_kwargs: keyword arguments to pass to `operator` when the waveform is sampled.
        """
        super().__init__()
        self.operator = operator
        self.operands = operands
        self.operator_kwargs = operator_kwargs

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        sampled_operands = tuple(
            (
                operand.sample_arr(time_array)
                if isinstance(operand, Waveform)
                else operand
            )
            for operand in self.operands
        )
        result = self.operator(*sampled_operands, **self.operator_kwargs)
        if not isinstance(result, np.ndarray):
            raise TypeError("Operator did not return an array")
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, OperationWaveform)
            and type(self) is type(other)
            and self.operator == other.operator
            and self.operands == other.operands
            and self.operator_kwargs == other.operator_kwargs
        )

    @override
    def __repr__(self) -> str:
        operator_repr = self.operator.__name__
        if isinstance(self.operator, np.ufunc):
            operator_repr = f"numpy.{operator_repr}"
            if self.operands and not self.operator_kwargs:
                if len(self.operands) == 1:
                    return f"{operator_repr}({repr(self.operands[0])})"
                return f"{operator_repr}{repr(self.operands)}"

        class_name = type(self).__name__
        operands_repr = ""
        for operand in self.operands:
            operands_repr += f", {repr(operand)}"
        operator_kwargs_repr = "".join(
            [
                _strings.param_repr(key, value)
                for key, value in self.operator_kwargs.items()
            ]
        )
        return f"{class_name}({operator_repr}{operands_repr}{operator_kwargs_repr})"
