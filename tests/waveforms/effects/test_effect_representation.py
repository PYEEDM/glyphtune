"""Tests for effect representation."""

from typing import override
from glyphtune import arrays, waveforms
from glyphtune.waveforms import effects


class DummyWaveform(waveforms.Waveform):
    """Useless waveform with a constant repr function."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        return super().sample_arr(time_array)

    @override
    def __repr__(self) -> str:
        return "DummyWaveform"


def test_effect_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    effect = effects.Effect(dummy_waveform)

    assert repr(effect) == f"Effect({dummy_waveform_repr})"


def test_effect_repr_with_mix_argument() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    effect = effects.Effect(dummy_waveform, mix=0.8)

    assert repr(effect) == f"Effect({dummy_waveform_repr}, mix=0.8)"


def test_stereo_pan_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    effect = effects.StereoPan(dummy_waveform, 0.8)

    assert repr(effect) == f"StereoPan({dummy_waveform_repr}, pan=0.8)"


def test_stereo_levels_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    effect = effects.StereoLevels(dummy_waveform, 0.8, mix=0.5)

    assert (
        repr(effect) == f"StereoLevels({dummy_waveform_repr}, left_level=0.8, mix=0.5)"
    )


def test_stereo_inter_mix_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    effect = effects.StereoInterMix(dummy_waveform, left_to_right=0.2)

    assert repr(effect) == f"StereoInterMix({dummy_waveform_repr}, left_to_right=0.2)"


def test_stereo_delay_repr() -> None:
    """Ensure the representation of the waveform is as expected."""
    dummy_waveform = DummyWaveform()
    dummy_waveform_repr = repr(dummy_waveform)
    effect = effects.StereoDelay(dummy_waveform, left_right_delay=0.1)

    assert repr(effect) == f"StereoDelay({dummy_waveform_repr}, left_right_delay=0.1)"
