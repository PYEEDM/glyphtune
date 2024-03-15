"""Audio signal representation and manipulation."""

from __future__ import annotations
from typing import Any, Literal, override
import numpy as np
import numpy.typing as npt


class Signal(np.lib.mixins.NDArrayOperatorsMixin):
    """Class representing audio signals.

    Basically a wrapper around a Numpy array with validation, properties, and manipulation methods
    specific to audio signals. Can be passed to Python operators and Numpy functions in most ways
    as long as a valid audio signal can be returned.
    """

    def __init__(self, data: npt.ArrayLike) -> None:
        """Initializes an audio signal with an array-like.

        Args:
            data: numerical array-like of the shape (channels, samples).
        """
        self.array = np.asarray(data)

    @property
    def array(self) -> np.ndarray[Any, np.dtype[np.float_]]:
        """Array containing the data of the signal."""
        return self.__array

    @array.setter
    def array(self, value: np.ndarray[Any, np.dtype[np.float_]]) -> None:
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError("Signal array must have a numeric type")
        if len(value.shape) != 2:
            raise ValueError("Signal must be of the shape (channels, samples)")
        self.__array = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the signal array."""
        return self.array.shape

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
    def absolute_peak(self) -> float:
        """Equivalent to `max(abs(signal.array))`."""
        result = max(abs(self.array))
        assert isinstance(result, float)
        return result

    @property
    def dc_offset(self) -> np.ndarray[Any, np.dtype[np.float_]]:
        """DC offset (also known as DC bias) along each channel of the signal."""
        result: np.ndarray[Any, np.dtype[np.float_]] = self.array.mean(axis=1)
        return result

    def normalize(self) -> None:
        """Normalizes the signal between -1 and 1 in-place."""
        self.array /= self.absolute_peak

    def remove_dc_offset(self) -> None:
        """Removes DC offset of the signal in-place."""
        self.array -= self.dc_offset

    def reverse(self) -> None:
        """Reverses the signal in-place."""
        self.array = np.flip(self.array, axis=1)

    @override
    def __eq__(self, other: Any) -> Any:
        return (
            isinstance(other, Signal)
            and type(self) is type(other)
            and np.array_equal(self, other)
        )

    @override
    def __repr__(self) -> str:
        return f"Signal(numpy.{repr(self.array)})"

    @override
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
        ],
        *inputs: Any,
        **kwargs: Any,
    ) -> Signal:
        converted_inputs = (np.asarray(operand) for operand in inputs)
        return Signal(
            self.array.__array_ufunc__(ufunc, method, *converted_inputs, **kwargs)
        )

    # TODO: use PEP 695 generics when black and mypy support them...
    # see https://github.com/psf/black/issues/4071 for black
    # see https://github.com/python/mypy/issues/15238 for mypy
    def __array__(
        self, dtype: np.dtype[Any] | None = None
    ) -> np.ndarray[Any, np.dtype[Any]]:
        return self.array.__array__(dtype)

    def __getitem__(self, key: Any) -> np.ndarray[Any, np.dtype[np.float_]] | float:
        result = self.array.__getitem__(key)
        assert isinstance(result, (np.ndarray, float))
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        # unfortunately, ndarray.__setitem__ is untyped...
        self.array.__setitem__(key, value)  # type: ignore[no-untyped-call]
