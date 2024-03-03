"""glyphtune is in pre-alpha, and so is this documentation."""

from typing import Any, TypeAlias
import numpy as np


# TODO: change to type statement when/if https://github.com/python/cpython/issues/114159 is resolved
FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float_]]
