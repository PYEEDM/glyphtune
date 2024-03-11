"""Convenience functionalities for string representations and operations."""

from typing import Any


def optional_param_repr(
    name: str, default_value: Any, value: Any, first: bool = False
) -> str:
    """Returns the representation of an optional parameter value if necessary.

    Args:
        name: the name of the parameter.
        default_value: the default value of the parameter.
        value: the actual value of the parameter.
    """
    prefix = "" if first else ", "
    return f"{prefix}{name}={value}" if value != default_value else ""
