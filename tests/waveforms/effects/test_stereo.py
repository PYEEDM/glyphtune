"""Tests for stereo effects module."""

from typing import override
import numpy as np
from glyphtune import signal, waveforms
from glyphtune.waveforms import effects


class ModifiedTimeWaveform(waveforms.Waveform):
    """Useless waveform that returns the time signal with the right channel offset by 1."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        result = signal.Signal(np.copy(time))
        result[1] += 1
        return result


def test_stereo_pan_zero_pan_is_equal_to_dry_signal() -> None:
    """Ensure that a pan effect with a pan value of 0 does nothing to the input."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_pan = effects.StereoPan(modified_time_waveform, 0)
    stereo_pan_signal = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    assert np.array_equal(stereo_pan_signal, reference_signal)


def test_stereo_pan_halfway_right() -> None:
    """Ensure correct result for a halfway right stereo pan."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_pan = effects.StereoPan(modified_time_waveform, 0.5)
    stereo_pan_signal = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[1] += 0.5 * reference_signal[0]
    reference_signal[0] *= 0.5
    assert np.array_equal(stereo_pan_signal, reference_signal)


def test_stereo_pan_completely_right() -> None:
    """Ensure correct result for a completely right stereo pan."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_pan = effects.StereoPan(modified_time_waveform, 1)
    stereo_pan_signal = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[1] += reference_signal[0]
    reference_signal[0] *= 0
    assert np.array_equal(stereo_pan_signal, reference_signal)


def test_stereo_pan_halfway_left() -> None:
    """Ensure correct result for a halfway right left pan."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_pan = effects.StereoPan(modified_time_waveform, -0.5)
    stereo_pan_signal = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[0] += 0.5 * reference_signal[1]
    reference_signal[1] *= 0.5
    assert np.array_equal(stereo_pan_signal, reference_signal)


def test_stereo_pan_completely_left() -> None:
    """Ensure correct result for a completely left stereo pan."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_pan = effects.StereoPan(modified_time_waveform, -1)
    stereo_pan_signal = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[0] += reference_signal[1]
    reference_signal[1] *= 0
    assert np.array_equal(stereo_pan_signal, reference_signal)


def test_stereo_levels() -> None:
    """Ensure correct result for adjusted stereo levels."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_levels = effects.StereoLevels(modified_time_waveform, 0.5, -0.5)
    stereo_levels_signal = stereo_levels.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[0] *= 0.5
    reference_signal[1] *= -0.5
    assert np.array_equal(stereo_levels_signal, reference_signal)


def test_stereo_inter_mix() -> None:
    """Ensure correct result for a stereo inter-mix."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_inter_mix = effects.StereoInterMix(modified_time_waveform, 0.5, -0.5, 1)
    stereo_inter_mix_signal = stereo_inter_mix.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal = signal.Signal(np.flip(reference_signal, axis=0))
    reference_signal[0] *= 0.5
    reference_signal[1] *= -0.5
    assert np.array_equal(stereo_inter_mix_signal, reference_signal)


def test_stereo_delay_positive_left_right_delay() -> None:
    """Ensure correct result for a positive left to right stereo delay."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_delay = effects.StereoDelay(modified_time_waveform, 0.5)
    stereo_delay_signal = stereo_delay.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[1] -= 0.5
    assert np.allclose(stereo_delay_signal, reference_signal)


def test_stereo_delay_negative_left_right_delay() -> None:
    """Ensure correct result for a negative left to right stereo delay."""
    sampling_rate = 10
    duration = 1
    modified_time_waveform = ModifiedTimeWaveform()

    stereo_delay = effects.StereoDelay(modified_time_waveform, -0.5)
    stereo_delay_signal = stereo_delay.sample_seconds(sampling_rate, duration)
    reference_signal = modified_time_waveform.sample_seconds(sampling_rate, duration)
    reference_signal[0] -= 0.5
    assert np.allclose(stereo_delay_signal, reference_signal)
