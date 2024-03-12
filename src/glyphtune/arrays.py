"""Convenience functionalities for arrays."""

from typing import Any, TypeAlias
import numpy as np


# TODO: change to type statement when/if https://github.com/python/cpython/issues/114159 is resolved
FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float_]]


def is_stereo_signal(signal: FloatArray) -> bool:
    """Returns whether the given signal is stereo.

    Args:
        signal: input signal array.
    """
    return len(signal.shape) == 2 and signal.shape[0] == 2
