"""Convenience functionalities for string representations and operations."""

from typing import Any


def param_repr(name: str, value: Any, first: bool = False) -> str:
    """Returns the representation of a parameter value.

    Args:
        name: name of the parameter.
        value: the value of the parameter.
        first: whether the parameter is the first one being passed.
    """
    prefix = "" if first else ", "
    return f"{prefix}{name}={value}"


def optional_param_repr(
    name: str, default_value: Any, value: Any, first: bool = False
) -> str:
    """Returns the representation of an optional parameter value if necessary.

    Args:
        name: the name of the parameter.
        default_value: the default value of the parameter.
        value: the actual value of the parameter.
        first: whether the parameter is the first one being passed.
    """
    return param_repr(name, value, first) if value != default_value else ""
