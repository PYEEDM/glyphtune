"""Tests for resampling module."""

from typing import override
import numpy as np
import pytest
from glyphtune import signal, waveforms


class TimeWaveform(waveforms.Waveform):
    """Useless waveform that just returns the time signal when sampled."""

    @override
    def sample_time(self, time: signal.Signal) -> signal.Signal:
        return time


def test_resampling_signal_with_original_sampling_rate_gives_same_signal() -> None:
    """Ensure resampling a signal using its original sampling rate returns the exact same signal."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)
    resampled_signal = resample_waveform.sample_seconds(duration, sampling_rate)

    assert np.array_equal(resampled_signal, sig)


def test_resampling_signal_with_half_sampling_rate_gives_even_samples() -> None:
    """Ensure resampling a signal with half its original sampling rate gives every other sample."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3, 4]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)
    resampled_signal = resample_waveform.sample_seconds(duration, sampling_rate // 2)

    assert np.array_equal(resampled_signal, signal.Signal([[0, 2]] * 2))


def test_resampling_signal_with_double_sampling_rate_gives_repeated_samples() -> None:
    """Ensure resampling a signal with double its original sampling rate gives repeated samples."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3, 4]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)
    resampled_signal = resample_waveform.sample_seconds(duration, sampling_rate * 2)

    assert np.array_equal(
        resampled_signal, signal.Signal([[0, 0, 1, 2, 2, 2, 3, 4]] * 2)
    )


def test_resampling_mono_signal_into_stereo_signal() -> None:
    """Ensure resampling mono signal into a stereo signal tiles the signal properly."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3]])

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)
    resampled_signal = resample_waveform.sample_seconds(duration, sampling_rate)

    assert np.array_equal(resampled_signal, signal.Signal([[0, 1, 2, 3]] * 2))


def test_resampling_stereo_signal_into_mono_signal() -> None:
    """Ensure resampling stereo signal into a mono signal returns the mean along each sample."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[1, -2, 3, -1], [1, 5, -7, -1]])

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)
    resampled_signal = resample_waveform.sample_seconds(
        duration, sampling_rate, channels=1
    )

    assert np.array_equal(resampled_signal, signal.Signal([[1, 3 / 2, -2, -1]]))


def test_resampling_from_two_channels_to_three_channels_raises_valueerror() -> None:
    """Ensure resampling from two channels to three channels raises a ValueError."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)

    with pytest.raises(ValueError):
        resample_waveform.sample_seconds(duration, sampling_rate, channels=3)


def test_resampling_from_three_channels_to_two_channels_raises_valueerror() -> None:
    """Ensure resampling from three channels to two channels raises a ValueError."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3]] * 3)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)

    with pytest.raises(ValueError):
        resample_waveform.sample_seconds(duration, sampling_rate)


def test_time_multiplier_same_as_multiplying_sampling_rate_positive_values() -> None:
    """Ensure a positive time multiplier is the same as multiplying original sampling rate."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3, 4]] * 2)

    resample_waveform_multiplier = waveforms.ResampleWaveform(sig, sampling_rate, 2)
    resampled_signal = resample_waveform_multiplier.sample_seconds(
        duration, sampling_rate
    )
    resample_waveform_sampling_rate = waveforms.ResampleWaveform(sig, sampling_rate * 2)
    resampled_signal_reference = resample_waveform_sampling_rate.sample_seconds(
        duration, sampling_rate
    )

    assert np.array_equal(resampled_signal, resampled_signal_reference)


def test_negative_time_multiplier_same_as_reversed_signal_with_positive_one() -> None:
    """Ensure a negative time multiplier is the same as reversing the signal with a positive one."""
    sampling_rate = 4
    duration = 1
    sig = signal.Signal([[0, 1, 2, 3, 4]] * 2)

    resample_waveform_multiplier = waveforms.ResampleWaveform(sig, sampling_rate, -1.3)
    resampled_signal = resample_waveform_multiplier.sample_seconds(
        duration, sampling_rate
    )
    resample_waveform_sampling_rate = waveforms.ResampleWaveform(
        sig.reverse(), sampling_rate, 1.3
    )
    resampled_signal_reference = resample_waveform_sampling_rate.sample_seconds(
        duration, sampling_rate
    )

    assert np.array_equal(resampled_signal, resampled_signal_reference)


def test_resampling_pads_with_zeros_for_time_values_with_no_data() -> None:
    """Ensure resampling pads samples outside the data with zeros."""
    sampling_rate = 4
    duration = 1
    padding_duration = 1
    sig = signal.Signal([[0, 1, 2, 3]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate)
    resampled_signal = resample_waveform.sample_seconds(
        duration + padding_duration * 2, sampling_rate, -padding_duration
    )

    padding = [0] * padding_duration * sampling_rate
    assert np.array_equal(
        resampled_signal, signal.Signal([padding + [0, 1, 2, 3] + padding] * 2)
    )


def test_non_default_resampling_pads_with_zeros_for_time_values_with_no_data() -> None:
    """Ensure resampling with non-default arguments pads samples outside the data with zeros."""
    sampling_rate = 4
    duration = 1
    padding_duration = 1
    sig = signal.Signal([[0, 1, 2, 3]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate * 2, -1)
    resampled_signal = resample_waveform.sample_seconds(
        duration + padding_duration * 2, sampling_rate, -padding_duration
    )

    padding = [0] * padding_duration * sampling_rate
    assert np.array_equal(
        resampled_signal, signal.Signal([padding + [3, 1, 0, 0] + padding] * 2)
    )


def test_resampling_signal_with_looping() -> None:
    """Ensure resampling a signal with looping gives expected results."""
    sampling_rate = 4
    duration = 10
    sig = signal.Signal([[0, 1, 2, 3]] * 2)

    resample_waveform = waveforms.ResampleWaveform(sig, sampling_rate, loop=True)
    resampled_signal = resample_waveform.sample_seconds(duration, sampling_rate)

    assert np.array_equal(
        resampled_signal, signal.Signal([[0, 1, 2, 3] * duration] * 2)
    )
