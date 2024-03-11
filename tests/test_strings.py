"""Tests for the strings internal module."""

from glyphtune import _strings


def test_optional_param_repr() -> None:
    """Ensure the representation of optional parameters is as expected."""
    assert _strings.optional_param_repr("test", 0, 1) == ", test=1"


def test_first_optional_param_repr() -> None:
    """Ensure the representation of optional parameters is as expected."""
    assert _strings.optional_param_repr("test", 0, 1, True) == "test=1"


def test_optional_param_repr_with_default_value() -> None:
    """Ensure the representation of optional parameters is empty string when using default value."""
    assert _strings.optional_param_repr("test", 0, 0) == ""


def test_first_optional_param_repr_with_default_value() -> None:
    """Ensure the representation of optional parameters is empty string when using default value."""
    assert _strings.optional_param_repr("test", 0, 0, True) == ""
