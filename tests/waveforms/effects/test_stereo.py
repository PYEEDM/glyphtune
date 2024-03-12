"""Tests for stereo effects module."""

from typing import override
import numpy as np
import pytest
from glyphtune import arrays, waveforms
from glyphtune.waveforms import effects


class ModifiedTimeArrayWaveform(waveforms.Waveform):
    """Useless waveform that returns the time array with the right channel offset by 1."""

    @override
    def sample_arr(self, time_array: arrays.FloatArray) -> arrays.FloatArray:
        result = time_array.copy()
        result[1] += 1
        return result


def test_stereo_pan_zero_pan_is_equal_to_dry_signal() -> None:
    """Ensure that a pan effect with a pan value of 0 does nothing to the input."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_pan = effects.StereoPan(modified_time_array_waveform, 0)
    sampled_stereo_pan = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    assert np.array_equal(sampled_stereo_pan, reference_samples)


def test_stereo_pan_halfway_right() -> None:
    """Ensure correct result for a halfway right stereo pan."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_pan = effects.StereoPan(modified_time_array_waveform, 0.5)
    sampled_stereo_pan = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[1] += 0.5 * reference_samples[0]
    reference_samples[0] *= 0.5
    assert np.array_equal(sampled_stereo_pan, reference_samples)


def test_stereo_pan_completely_right() -> None:
    """Ensure correct result for a completely right stereo pan."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_pan = effects.StereoPan(modified_time_array_waveform, 1)
    sampled_stereo_pan = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[1] += reference_samples[0]
    reference_samples[0] *= 0
    assert np.array_equal(sampled_stereo_pan, reference_samples)


def test_stereo_pan_halfway_left() -> None:
    """Ensure correct result for a halfway right left pan."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_pan = effects.StereoPan(modified_time_array_waveform, -0.5)
    sampled_stereo_pan = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[0] += 0.5 * reference_samples[1]
    reference_samples[1] *= 0.5
    assert np.array_equal(sampled_stereo_pan, reference_samples)


def test_stereo_pan_completely_left() -> None:
    """Ensure correct result for a completely left stereo pan."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_pan = effects.StereoPan(modified_time_array_waveform, -1)
    sampled_stereo_pan = stereo_pan.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[0] += reference_samples[1]
    reference_samples[1] *= 0
    assert np.array_equal(sampled_stereo_pan, reference_samples)


def test_stereo_levels() -> None:
    """Ensure correct result for adjusted stereo levels."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_levels = effects.StereoLevels(modified_time_array_waveform, 0.5, -0.5)
    sampled_stereo_levels = stereo_levels.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[0] *= 0.5
    reference_samples[1] *= -0.5
    assert np.array_equal(sampled_stereo_levels, reference_samples)


def test_stereo_inter_mix() -> None:
    """Ensure correct result for a stereo inter-mix."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_inter_mix = effects.StereoInterMix(
        modified_time_array_waveform, 0.5, -0.5, 1
    )
    sampled_stereo_inter_mix = stereo_inter_mix.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples = np.flip(reference_samples, axis=0)
    reference_samples[0] *= 0.5
    reference_samples[1] *= -0.5
    assert np.array_equal(sampled_stereo_inter_mix, reference_samples)


def test_stereo_delay_positive_left_right_delay() -> None:
    """Ensure correct result for a positive left to right stereo delay."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_delay = effects.StereoDelay(modified_time_array_waveform, 0.5)
    sampled_stereo_delay = stereo_delay.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[1] -= 0.5
    assert sampled_stereo_delay == pytest.approx(reference_samples)


def test_stereo_delay_negative_left_right_delay() -> None:
    """Ensure correct result for a negative left to right stereo delay."""
    sampling_rate = 10
    duration = 1
    modified_time_array_waveform = ModifiedTimeArrayWaveform()

    stereo_delay = effects.StereoDelay(modified_time_array_waveform, -0.5)
    sampled_stereo_delay = stereo_delay.sample_seconds(sampling_rate, duration)
    reference_samples = modified_time_array_waveform.sample_seconds(
        sampling_rate, duration
    )
    reference_samples[0] -= 0.5
    assert sampled_stereo_delay == pytest.approx(reference_samples)
