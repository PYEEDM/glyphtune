"""Tests for arrays module."""

import numpy as np
from glyphtune import arrays


def test_is_stereo_signal_with_stereo_signal() -> None:
    """Ensure `is_stereo_signal` returns `True` when given valid stereo signal."""
    signal = np.array([[1, 2, 3, 4], [1, 3, 4, 7]])

    assert arrays.is_stereo_signal(signal)


def test_is_stereo_signal_with_mono_signal() -> None:
    """Ensure `is_stereo_signal` returns `False` when given mono signal."""
    signal = np.array([[1, 2, 3, 4]])

    assert not arrays.is_stereo_signal(signal)


def test_is_stereo_signal_with_three_channel_signal() -> None:
    """Ensure `is_stereo_signal` returns `False` when given signal with three channels."""
    signal = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    assert not arrays.is_stereo_signal(signal)


def test_is_stereo_signal_with_1d_signal() -> None:
    """Ensure `is_stereo_signal` returns `False` when given a 1d array."""
    signal = np.array([1, 2, 3, 4])

    assert not arrays.is_stereo_signal(signal)


def test_is_stereo_signal_with_3d_signal() -> None:
    """Ensure `is_stereo_signal` returns `False` when given a 3d array."""
    signal = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])

    assert not arrays.is_stereo_signal(signal)
