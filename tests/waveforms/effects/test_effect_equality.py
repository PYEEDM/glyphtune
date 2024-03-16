"""Tests for effect equality checks."""

from typing import Any, override
from glyphtune import signal, waveforms
from glyphtune.waveforms import effects


class DummyWaveform(waveforms.Waveform):
    """Useless waveform that's always equal to any other waveform of its kind."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return super().sample_time(time)

    @override
    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other)


def test_effect_equality() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    effect = effects.Effect(dummy_waveform, 0.2)
    also_effect = effects.Effect(dummy_waveform, 0.2)
    assert effect == also_effect


def test_effect_inequality_type() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    effect = effects.Effect(dummy_waveform, 0.2)
    not_effect = DummyWaveform()
    assert effect != not_effect


def test_effect_inequality_input_waveform() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    effect = effects.Effect(dummy_waveform, 0.2)
    other_effect = effects.Effect(-dummy_waveform, 0.2)
    assert effect != other_effect


def test_effect_inequality_mix() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    effect = effects.Effect(dummy_waveform, 0.2)
    other_effect = effects.Effect(dummy_waveform, 0.3)
    assert effect != other_effect


def test_stereo_pan_equality() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_pan = effects.StereoPan(dummy_waveform, 0.2)
    also_stereo_pan = effects.StereoPan(dummy_waveform, 0.2)
    assert stereo_pan == also_stereo_pan


def test_stereo_pan_inequality_pan() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_pan = effects.StereoPan(dummy_waveform, 0.2)
    other_stereo_pan = effects.StereoPan(dummy_waveform, 1)
    assert stereo_pan != other_stereo_pan


def test_stereo_levels_equality() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_levels = effects.StereoLevels(dummy_waveform, 0.2, 0.4)
    also_stereo_levels = effects.StereoLevels(dummy_waveform, 0.2, 0.4)
    assert stereo_levels == also_stereo_levels


def test_stereo_levels_inequality_left_level() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_levels = effects.StereoLevels(dummy_waveform, 0.2)
    other_stereo_levels = effects.StereoLevels(dummy_waveform, 1)
    assert stereo_levels != other_stereo_levels


def test_stereo_levels_inequality_right_level() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_levels = effects.StereoLevels(dummy_waveform, right_level=0.2)
    other_stereo_levels = effects.StereoLevels(dummy_waveform, right_level=1)
    assert stereo_levels != other_stereo_levels


def test_stereo_inter_mix_equality() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_inter_mix = effects.StereoInterMix(dummy_waveform, 0.2, 0.4)
    also_stereo_inter_mix = effects.StereoInterMix(dummy_waveform, 0.2, 0.4)
    assert stereo_inter_mix == also_stereo_inter_mix


def test_stereo_inter_mix_inequality_right_to_left() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_inter_mix = effects.StereoInterMix(dummy_waveform, 0.2)
    other_stereo_inter_mix = effects.StereoInterMix(dummy_waveform, 1)
    assert stereo_inter_mix != other_stereo_inter_mix


def test_stereo_inter_mix_inequality_left_to_right() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_inter_mix = effects.StereoInterMix(dummy_waveform, left_to_right=0.2)
    other_stereo_inter_mix = effects.StereoInterMix(dummy_waveform, left_to_right=1)
    assert stereo_inter_mix != other_stereo_inter_mix


def test_stereo_delay_equality() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_delay = effects.StereoDelay(dummy_waveform, 0.2)
    also_stereo_delay = effects.StereoDelay(dummy_waveform, 0.2)
    assert stereo_delay == also_stereo_delay


def test_stereo_delay_inequality_left_right_delay() -> None:
    """Ensure equality check result is as expected."""
    dummy_waveform = DummyWaveform()
    stereo_delay = effects.StereoDelay(dummy_waveform, 0.2)
    other_stereo_delay = effects.StereoDelay(dummy_waveform, 1)
    assert stereo_delay != other_stereo_delay
