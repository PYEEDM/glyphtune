"""Tests for signal module."""

import numpy as np
import pytest
from glyphtune import signal


def test_is_signal_like_with_numpy_array() -> None:
    """Ensure `is_signal_like` returns True with valid numpy array."""
    data = np.array([[1, 2, 3]])

    assert signal.is_signal_like(data)


def test_is_signal_like_with_list() -> None:
    """Ensure `is_signal_like` returns True with valid list."""
    data = [[1, 2, 3]]

    assert signal.is_signal_like(data)


def test_is_signal_like_with_numpy_array_wrong_type() -> None:
    """Ensure `is_signal_like` returns False with numpy array of a bad type."""
    data = np.array([[3j + 3, 4j - 2]])

    assert not signal.is_signal_like(data)


def test_is_signal_like_with_list_wrong_type() -> None:
    """Ensure `is_signal_like` returns False with list of a bad type."""
    data = [[False, True, True]]

    assert not signal.is_signal_like(data)


def test_is_signal_like_with_numpy_array_wrong_shape() -> None:
    """Ensure `is_signal_like` returns False with numpy array of a bad shape."""
    data = np.array([1, 2, 3])

    assert not signal.is_signal_like(data)


def test_is_signal_like_with_list_wrong_shape() -> None:
    """Ensure `is_signal_like` returns False with list of a bad shape."""
    data = [[[1, 2, 3]]]

    assert not signal.is_signal_like(data)


def test_signal_init_with_numpy_array_uses_the_same_array() -> None:
    """Ensure signals are initialized with the same array they are given."""
    data = np.array([[1, 2, 3]])

    sig = signal.Signal(data)

    assert sig.array is data


def test_signal_init_with_signal_uses_the_same_array() -> None:
    """Ensure signals are initialized with the same array of the signal they are given."""
    data = np.array([[1, 2, 3]])

    sig = signal.Signal(signal.Signal(data))

    assert sig.array is data


def test_signal_init_with_list_is_same_as_numpy_array_init_with_list() -> None:
    """Ensure array of signal initialized with list is same as array initialized with that list."""
    data = [[1, 2, 3]]

    sig = signal.Signal(signal.Signal(data))

    assert np.array_equal(sig.array, np.array(data))


def test_signal_init_with_numpy_array_of_wrong_type_raises_valueerror() -> None:
    """Ensure signal initialized with an array of a bad type raises a ValueError."""
    data = np.array([[3j + 3, 4j - 2]])

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_signal_init_with_list_of_wrong_type_raises_valueerror() -> None:
    """Ensure signal initialized with a list of a bad type raises a ValueError."""
    data = [[False, True, True]]

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_signal_init_with_numpy_array_of_wrong_shape_raises_valueerror() -> None:
    """Ensure signal initialized with an array of a bad shape raises a ValueError."""
    data = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_signal_init_with_list_of_wrong_shape_raises_valueerror() -> None:
    """Ensure signal initialized with a list of a bad shape raises a ValueError."""
    data = [[[1, 2, 3]]]

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_channel_count() -> None:
    """Ensure signal channel count calculation is as expected."""
    data = np.array([[1, 2, 3], [1, 5, 7]])

    sig = signal.Signal(data)

    assert sig.channels == 2


def test_length() -> None:
    """Ensure signal sample length calculation is as expected."""
    data = np.array([[1, 2, 3], [1, 5, 7]])

    sig = signal.Signal(data)

    assert sig.length == 3


def test_absolute_peak() -> None:
    """Ensure signal absolute peak calculation is as expected."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert sig.absolute_peak == 7


def test_dc_offset() -> None:
    """Ensure signal DC offset calculation is as expected."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(sig.dc_offset, [2 / 3, -1 / 3])


def test_normalize() -> None:
    """Ensure signal normalization calculation is as expected."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    normalized_reference = data / 7
    assert np.array_equal(sig.normalize().array, normalized_reference)


def test_normalize_makes_absolute_peak_one() -> None:
    """Ensure signal normalization returns a signal whose absolute peak is 1."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert sig.normalize().absolute_peak == 1


def test_remove_dc_offset() -> None:
    """Ensure signal DC offset removal calculation is as expected."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.allclose(
        sig.remove_dc_offset().array, [[1 / 3, -8 / 3, 7 / 3], [4 / 3, 16 / 3, -20 / 3]]
    )


def test_remove_dc_offset_makes_dc_offset_zero() -> None:
    """Ensure signal DC offset removal returns a signal whose DC offset is 0."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.allclose(sig.remove_dc_offset().dc_offset, 0)


def test_reverse() -> None:
    """Ensure signal reversal calculation is as expected."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    reverse_data_reference = np.array([[3, -2, 1], [-7, 5, 1]])
    assert np.array_equal(sig.reverse().array, reverse_data_reference)


def test_repr() -> None:
    """Ensure signal representation is as expected."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert repr(sig) == f"Signal(numpy.{repr(data)})"


def test_list_init_repr() -> None:
    """Ensure signal representation is as expected if initialized with list."""
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert repr(sig) == f"Signal(numpy.{repr(np.array(data))})"


def test_addition_with_scalar_results_in_signal() -> None:
    """Ensure an addition operation between a signal and a scalar results in a signal."""
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert isinstance(sig + 1, signal.Signal)


def test_addition_with_signal_results_in_signal() -> None:
    """Ensure an addition operation between two signals results in a signal."""
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert isinstance(sig + sig, signal.Signal)


def test_diff_on_signal_results_in_signal() -> None:
    """Ensure a Numpy diff operation on a signal results in a signal."""
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert isinstance(np.diff(sig), signal.Signal)


def test_max_does_not_result_in_signal() -> None:
    """Ensure a Numpy max operation on a signal does not result in a signal."""
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert not isinstance(np.max(sig), signal.Signal)


def test_expand_dims_does_not_result_in_signal() -> None:
    """Ensure a Numpy expand_dims operation on a signal does not result in a signal."""
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert not isinstance(np.expand_dims(sig, axis=0), signal.Signal)


def test_signal_conversion_to_array_returns_signal_array() -> None:
    """Ensure converting a signal to array returns an array equal to the signal's array."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(np.asarray(sig), data)


def test_signal_conversion_to_array_with_dtype() -> None:
    """Ensure converting signal to array with type equals converting signal's array to that type."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(np.asarray(sig, float), data.astype(float))


def test_signal_single_indexing() -> None:
    """Ensure signal indexing works just like array indexing."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(sig[0], data[0])


def test_signal_nested_indexing() -> None:
    """Ensure signal indexing works just like array indexing."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(sig[0, 0], data[0, 0])


def test_signal_range_indexing() -> None:
    """Ensure signal indexing works just like array indexing."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(sig[0, :1], data[0, :1])


def test_signal_array_indexing() -> None:
    """Ensure signal indexing works just like array indexing."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(sig[sig > 2], data[data > 2])


def test_signal_single_item_assignment() -> None:
    """Ensure signal item assignment works just like array item assignment."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)
    sig[0] = 2

    data[0] = 2
    assert np.array_equal(sig, data)


def test_signal_nested_item_assignment() -> None:
    """Ensure signal item assignment works just like array item assignment."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)
    sig[0, 0] = 2

    data[0, 0] = 2
    assert np.array_equal(sig, data)


def test_signal_range_item_assignment() -> None:
    """Ensure signal item assignment works just like array item assignment."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)
    sig[0, :1] = 2

    data[0, :1] = 2
    assert np.array_equal(sig, data)


def test_signal_array_item_assignment() -> None:
    """Ensure signal item assignment works just like array item assignment."""
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)
    sig[sig > 2] = 2

    data[data > 2] = 2
    assert np.array_equal(sig, data)
