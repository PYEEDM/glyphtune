import numpy as np
import pytest
from glyphtune import signal


def test_signal_init_with_numpy_array_uses_the_same_array() -> None:
    data = np.array([[1, 2, 3]])

    sig = signal.Signal(data)

    assert sig.array is data


def test_signal_init_with_signal_uses_the_same_array() -> None:
    data = np.array([[1, 2, 3]])

    sig = signal.Signal(signal.Signal(data))

    assert sig.array is data


def test_signal_init_with_list_is_same_as_numpy_array_init_with_list() -> None:
    data = [[1, 2, 3]]

    sig = signal.Signal(signal.Signal(data))

    assert np.array_equal(sig.array, np.array(data))


def test_signal_init_with_numpy_array_of_wrong_type_raises() -> None:
    data = np.array([[3j + 3, 4j - 2]])

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_signal_init_with_list_of_wrong_type_raises() -> None:
    data = [[False, True, True]]

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_signal_init_with_numpy_array_of_wrong_shape_raises() -> None:
    data = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_signal_init_with_list_of_wrong_shape_raises() -> None:
    data = [[[1, 2, 3]]]

    with pytest.raises(ValueError):
        signal.Signal(data)


def test_channel_count() -> None:
    data = np.array([[1, 2, 3], [1, 5, 7]])

    sig = signal.Signal(data)

    assert sig.channels == 2


def test_absolute_peak() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert sig.absolute_peak == 7


def test_dc_offset() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.array_equal(sig.dc_offset, [2 / 3, -1 / 3])


def test_normalize() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.max(np.abs(sig.normalize().array)) == 1


def test_normalize_makes_absolute_peak_one() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert sig.normalize().absolute_peak == 1


def test_remove_dc_offset() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.allclose(
        sig.remove_dc_offset().array, [[1 / 3, -8 / 3, 7 / 3], [4 / 3, 16 / 3, -20 / 3]]
    )


def test_remove_dc_offset_makes_dc_offset_zero() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert np.allclose(sig.remove_dc_offset().dc_offset, 0)


def test_reverse() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    reverse_data_reference = np.array([[3, -2, 1], [-7, 5, 1]])
    assert np.array_equal(sig.reverse().array, reverse_data_reference)


def test_repr() -> None:
    data = np.array([[1, -2, 3], [1, 5, -7]])

    sig = signal.Signal(data)

    assert repr(sig) == f"Signal(numpy.{repr(data)})"


def test_list_init_repr() -> None:
    data = [[1, -2, 3], [1, 5, -7]]

    sig = signal.Signal(data)

    assert repr(sig) == f"Signal(numpy.{repr(np.array(data))})"
