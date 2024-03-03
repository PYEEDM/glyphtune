"""Tests for differentiation and integral waveforms."""

import numpy as np
import pytest
from glyphtune import waveforms


def test_sine_derivative_is_cosine() -> None:
    """Ensure that the derivative of a sine wave is a cosine wave."""
    sampling_rate = 10**4
    duration = 1
    frequency = 7
    error_margin = 10**-2

    sine_wave = waveforms.Sine(frequency)
    sine_wave_derivative = waveforms.DerivativeWaveform(sine_wave)

    cosine_wave = waveforms.Sine(frequency, 0.25)
    sampled_derivative = sine_wave_derivative.sample_seconds(sampling_rate, duration)
    sampled_cosine = cosine_wave.sample_seconds(sampling_rate, duration)
    reference_samples = 2 * np.pi * frequency * sampled_cosine
    assert sampled_derivative == pytest.approx(reference_samples, abs=error_margin)


def test_sine_integral_is_minus_cosine() -> None:
    """Ensure that the integral of a sine wave is a negative cosine wave."""
    sampling_rate = 10**4
    duration = 1
    frequency = 4
    error_margin = 10**-2

    sine_wave = waveforms.Sine(frequency)
    c = -1 / (2 * np.pi * frequency)
    sine_wave_integral = waveforms.IntegralWaveform(sine_wave) + c

    cosine_wave = waveforms.Sine(frequency, 0.25)
    sampled_integral = sine_wave_integral.sample_seconds(sampling_rate, duration)
    sampled_cosine = cosine_wave.sample_seconds(sampling_rate, duration)
    reference_samples = -sampled_cosine / (2 * np.pi * frequency)
    assert sampled_integral == pytest.approx(reference_samples, abs=error_margin)


def test_stitching_dynamic_integral_does_not_affect_values() -> None:
    """Ensure sampling integral waveform is equal to stitching multiple sampled chunks of it."""
    sampling_rate = 10**4
    duration = 1
    frequency = 1.33
    chunk_count = 16
    chunk_duration = duration / chunk_count
    error_margin = 10**-2

    sine_wave = waveforms.Sine(frequency)
    sine_wave_integral = waveforms.IntegralWaveform(sine_wave, dynamic_offset=False)
    sampled_integral = sine_wave_integral.sample_seconds(sampling_rate, duration)

    sine_wave_integral = waveforms.IntegralWaveform(sine_wave, dynamic_offset=True)
    sampled_integral_stitched = np.zeros(0)
    for chunk in range(chunk_count):
        chunk_offset = chunk * chunk_duration
        sampled_chunk = sine_wave_integral.sample_seconds(
            sampling_rate, chunk_duration, chunk_offset
        )
        sampled_integral_stitched = np.concatenate(
            (sampled_integral_stitched, sampled_chunk)
        )
    assert sampled_integral == pytest.approx(
        sampled_integral_stitched, abs=error_margin
    )


def test_resetting_integral_offset() -> None:
    """Ensure sampling after resetting integral offset is equivalent to sampling a new integral."""
    sampling_rate = 100
    duration = 0.1
    frequency = 1

    sine_wave = waveforms.Sine(frequency)
    sine_wave_integral = waveforms.IntegralWaveform(sine_wave, dynamic_offset=True)
    sampled_integral = sine_wave_integral.sample_seconds(sampling_rate, duration)
    sine_wave_integral.reset_offset()
    sampled_integral_reset = sine_wave_integral.sample_seconds(sampling_rate, duration)

    assert sampled_integral == pytest.approx(sampled_integral_reset)


def test_integral_of_derivative_of_sine_equals_sine() -> None:
    """Ensure the derivative of an integral of a sine wave is equal to the sine wave itself."""
    sampling_rate = 10**4
    duration = 1
    error_margin = 10**-2
    sine_wave = waveforms.Sine(3.74)

    sine_wave_derivative = waveforms.DerivativeWaveform(sine_wave)
    integral_of_derivative = waveforms.IntegralWaveform(sine_wave_derivative)

    sampled_sine_wave = sine_wave.sample_seconds(sampling_rate, duration)
    sampled_integral = integral_of_derivative.sample_seconds(sampling_rate, duration)
    assert sampled_sine_wave == pytest.approx(sampled_integral, abs=error_margin)


def test_derivative_of_integral_of_sine_equals_sine() -> None:
    """Ensure the integral of a derivative of a sine wave is equal to the sine wave itself."""
    sampling_rate = 10**4
    duration = 1
    error_margin = 10**-2
    sine_wave = waveforms.Sine(3.74)

    sine_wave_integral = waveforms.IntegralWaveform(sine_wave)
    derivative_of_integral = waveforms.DerivativeWaveform(sine_wave_integral)

    sampled_sine_wave = sine_wave.sample_seconds(sampling_rate, duration)
    sampled_derivative = derivative_of_integral.sample_seconds(sampling_rate, duration)
    assert sampled_sine_wave == pytest.approx(sampled_derivative, abs=error_margin)
