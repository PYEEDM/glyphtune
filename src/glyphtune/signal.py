"""Audio signal representation and manipulation."""

from __future__ import annotations
from typing import Any, Callable, Iterable, Literal, Mapping, override
import numbers
import numpy as np
import numpy.typing as npt


class Signal(np.lib.mixins.NDArrayOperatorsMixin):
    """Class representing audio signals.

    Basically a wrapper around a Numpy array with validation, properties, and manipulation methods
    specific to audio signals. Can be indexed like Numpy arrays, and can be passed to Python
    operators and Numpy functions in most ways, which returns a new signal if possible,
    or, if no possible signal results from an operation, a normal Numpy array is returned instead.
    """

    def __init__(self, data: npt.ArrayLike) -> None:
        """Initializes an audio signal with an array-like.

        Args:
            data: numerical array-like of the shape (channels, samples).
        """
        self.data = np.asarray(data)

    @property
    def data(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Array containing the data of the signal."""
        return self.__data

    @data.setter
    def data(self, value: np.ndarray[Any, np.dtype[Any]]) -> None:
        if not _is_real(value.dtype):
            raise ValueError("Signal data array must have a real numeric type")
        if len(value.shape) != 2:
            raise ValueError(
                "Signal data array must be of the shape (channels, samples)"
            )
        self.__data = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the signal array."""
        return self.data.shape

    @property
    def channels(self) -> int:
        """The number of channels of the signal."""
        return self.shape[0]

    @property
    def length(self) -> int:
        """The length, in samples, of the signal."""
        return self.shape[1]

    @property
    def is_mono(self) -> bool:
        """Equivalent to `signal.channels == 1`."""
        return self.channels == 1

    @property
    def is_stereo(self) -> bool:
        """Equivalent to `signal.channels == 2`."""
        return self.channels == 2

    @property
    def absolute_peak(self) -> numbers.Real:
        """Equivalent to `max(abs(signal.data))`."""
        result = np.max(np.abs(self.data))
        assert isinstance(result, numbers.Real)
        return result

    @property
    def dc_offset(self) -> np.ndarray[Any, np.dtype[Any]]:
        """DC offset (also known as DC bias) along each channel of the signal."""
        result = self.data.mean(axis=1)
        assert isinstance(result, np.ndarray)
        return result

    def to_mono(self) -> Signal:
        """Returns the signal converted to mono.

        This is done by taking the mean along all channels of every sample.
        """
        return Signal(np.expand_dims(self.data.mean(axis=0), axis=0))

    def expand_channels(self, channels: int) -> Signal:
        """Returns the expansion of a mono signal into an arbitrary number of channels.

        This is done by tiling copies of the mono signal for each channel.
        """
        if not self.is_mono:
            raise ValueError("Can only expand channels of mono signal")
        if channels < 1:
            raise ValueError("Number of channels must be positive")
        return Signal(np.tile(self.data, (channels, 1)))

    def normalize(self) -> Signal:
        """Returns the signal normalized between -1 and 1."""
        return Signal(self.data / self.absolute_peak)

    def remove_dc_offset(self) -> Signal:
        """Returns the signal with the DC offset removed."""
        return Signal(self.data - np.expand_dims(self.dc_offset, axis=1))

    def reverse(self) -> Signal:
        """Returns the signal reversed."""
        return Signal(np.flip(self.data, axis=1))

    @override
    def __repr__(self) -> str:
        return f"Signal(numpy.{repr(self.data)})"

    @override
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
        ],
        *inputs: Any,
        **kwargs: Any,
    ) -> Signal | np.ndarray[Any, np.dtype[Any]]:
        converted_inputs = (np.asarray(operand) for operand in inputs)
        result_data = self.data.__array_ufunc__(
            ufunc, method, *converted_inputs, **kwargs
        )
        if is_signal_like(result_data):
            return Signal(result_data)
        return np.asarray(result_data)

    def __array_function__(
        self,
        func: Callable[..., Any],
        _types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        converted_args = tuple(np.asarray(arg) for arg in args)
        converted_types = tuple(type(arg) for arg in converted_args)
        result_data = self.data.__array_function__(
            func, converted_types, converted_args, kwargs
        )
        if is_signal_like(result_data):
            return Signal(result_data)
        return np.asarray(result_data)

    # TODO: use PEP 695 generics when black and mypy support them...
    # see https://github.com/psf/black/issues/4071 for black
    # see https://github.com/python/mypy/issues/15238 for mypy
    def __array__(
        self, dtype: np.dtype[Any] | None = None
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if dtype is None:
            dtype = self.data.dtype
        return self.data.__array__(dtype)

    def __getitem__(
        self, key: Any
    ) -> np.ndarray[Any, np.dtype[Any]] | np.float_ | np.int_:
        result = self.data[key]
        assert isinstance(result, (np.ndarray, np.float_, np.int_))
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value


def is_signal_like(data: npt.ArrayLike) -> bool:
    """Returns whether the given data can be a signal.

    Args:
        data: data to check.
    """
    array = np.asarray(data)
    return len(array.shape) == 2 and _is_real(array.dtype)


def _is_real(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
