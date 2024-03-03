"""Basic functionality of waveforms and operations on them.

See [`waveform.Waveform`][glyphtune.waveforms.Waveform]
and [`waveform.OperationWaveform`][glyphtune.waveforms.OperationWaveform]
for more information.
"""

from __future__ import annotations
from typing import Any, Callable, Literal, override
import copy
import numpy as np
import glyphtune


class Waveform(np.lib.mixins.NDArrayOperatorsMixin):
    """Base class representing waveforms.

    Waveform objects do not store any audio data. Instead, they must generate that data when
    one of their sampling functions is called.

    Waveform objects can be used as operands with Python operators and as inputs to NumPy functions
    along with other waveforms and numbers. This returns a
    [`waveforms.OperationWaveform`][glyphtune.waveforms.OperationWaveform].
    """

    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        """Returns an array containing the sampled signal of the waveform.

        The returned array will contain the same number of values as `time_array`.
        Each value in the returned array is the sample of the waveform's signal at the
        corresponding time value (in seconds) in `time_array`.

        Args:
            time_array: an array containing the values of the time variable at each sample point.
        """
        raise NotImplementedError(
            f"Attempted to sample a base {type(self).__name__} object"
        )

    def sample_seconds(
        self, sampling_rate: int, duration: float, start_offset: float = 0
    ) -> glyphtune.FloatArray:
        """Returns an array containing the sampled signal of the waveform.

        The returned array will contain `ceil(sampling_rate*duration)` samples.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            duration: the duration of time to be sampled in seconds.
            start_offset: the starting offset in seconds.
        """
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        end = start_offset + duration
        sampling_period = 1 / sampling_rate
        time_array = np.arange(start_offset, end, sampling_period)
        return self.sample_arr(time_array)

    def sample_samples(
        self, sampling_rate: int, count: int, start_offset: int = 0
    ) -> glyphtune.FloatArray:
        """Returns an array containing the sampled signal of the waveform.

        The returned array will contain `count` samples.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            count: the number of samples to take.
            start_offset: the starting offset in samples.
        """
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        duration_seconds = count / sampling_rate
        start_offset_seconds = start_offset / sampling_rate
        return self.sample_seconds(
            sampling_rate, duration_seconds, start_offset_seconds
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
            *operands: operands whose (sampled) output is passed to the operator when the waveform
                is sampled.
            **operator_kwargs: keyword arguments to pass to `operator` when the waveform is sampled.
        """
        super().__init__()
        self.__operator = operator
        self.__operands = operands
        self._operator_kwargs = operator_kwargs

    @property
    def operator(self) -> Callable[..., Any]:
        """The operator of this waveform."""
        return self.__operator

    @property
    def operands(self) -> tuple[Waveform | float, ...]:
        """A copy of the operands of this waveform."""
        return copy.copy(self.__operands)

    @property
    def operator_kwargs(self) -> dict[str, Any]:
        """A copy of the keyword arguments this waveform passes to its operator when sampled."""
        return copy.copy(self._operator_kwargs)

    @override
    def sample_arr(self, time_array: glyphtune.FloatArray) -> glyphtune.FloatArray:
        sampled_operands = tuple(
            (
                operand.sample_arr(time_array)
                if isinstance(operand, Waveform)
                else operand
            )
            for operand in self.__operands
        )
        result = self.__operator(*sampled_operands, **self._operator_kwargs)
        if not isinstance(result, np.ndarray):
            raise TypeError("Operator did not return an array")
        return result

    @override
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, OperationWaveform)
            and type(self) is type(other)
            and self.__operator == other.operator
            and self.__operands == other.operands
            and self._operator_kwargs == other.operator_kwargs
        )

    @override
    def __repr__(self) -> str:
        operator_repr = self.operator.__name__
        if isinstance(self.operator, np.ufunc):
            operator_repr = f"numpy.{operator_repr}"
            if self.__operands and not self.operator_kwargs:
                if len(self.__operands) == 1:
                    return f"{operator_repr}({repr(self.__operands[0])})"
                return f"{operator_repr}{repr(self.__operands)}"

        class_name = type(self).__name__
        operands_repr = ""
        for operand in self.__operands:
            operands_repr += f", {repr(operand)}"
        kwargs_repr = self._kwargs_repr()
        return f"{class_name}({operator_repr}{operands_repr}{kwargs_repr})"

    def _kwargs_repr(self) -> str:
        kwargs_repr = ""
        for key, value in self.operator_kwargs.items():
            kwargs_repr += f", {key}={repr(value)}"
        return kwargs_repr
