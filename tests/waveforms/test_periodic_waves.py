"""Tests for periodic waves."""

import numpy as np
import pytest
from glyphtune import arrays, waveforms


def _get_peak_frequency(signal: arrays.FloatArray, sampling_rate: int) -> float:
    """Return the peak frequency of the given signal's spectrum.

    Args:
        signal: signal to extract the peak frequency from.
        sampling_rate: the sampling rate of the signal in samples per second.
    """

    # Remove DC offset
    signal -= signal.mean()
    spectrum = np.abs(np.fft.fft(signal))
    frequencies = np.fft.fftfreq(len(spectrum))
    peak_frequency: float = abs(frequencies[np.argmax(spectrum)] * sampling_rate)
    return peak_frequency


def test_sine_shape_1hz() -> None:
    """Ensure the shape of a 1Hz sine wave is correct."""
    sine_waveform = waveforms.Sine(1)

    sampled_sine_waveform = sine_waveform.sample_samples(8, 9)

    reference_sine_samples = [0, 2**-0.5, 1, 2**-0.5, 0, -(2**-0.5), -1, -(2**-0.5), 0]
    assert sampled_sine_waveform == pytest.approx(reference_sine_samples)


def test_sine_shape_2hz() -> None:
    """Ensure the shape of a 2Hz sine wave is correct."""
    sine_waveform = waveforms.Sine(2)

    sampled_sine_waveform = sine_waveform.sample_samples(8, 9)
    reference_sine_samples = [0, 1, 0, -1, 0, 1, 0, -1, 0]
    assert sampled_sine_waveform == pytest.approx(reference_sine_samples)


def test_sawtooth_shape_1hz() -> None:
    """Ensure the shape of a 1Hz sawtooth wave is correct."""
    sawtooth_waveform = waveforms.Sawtooth(1)

    sampled_sawtooth_waveform = sawtooth_waveform.sample_samples(8, 9)
    reference_sawtooth_samples = [0, 0.25, 0.5, 0.75, -1, -0.75, -0.5, -0.25, 0]
    assert sampled_sawtooth_waveform == pytest.approx(reference_sawtooth_samples)


def test_sawtooth_shape_2hz() -> None:
    """Ensure the shape of a 2Hz sawtooth wave is correct."""
    sawtooth_waveform = waveforms.Sawtooth(2)

    sampled_sawtooth_waveform = sawtooth_waveform.sample_samples(8, 9)
    reference_sawtooth_samples = [0, 0.5, -1, -0.5, 0, 0.5, -1, -0.5, 0]
    assert sampled_sawtooth_waveform == pytest.approx(reference_sawtooth_samples)


def test_pulse_shape_1hz() -> None:
    """Ensure the shape of a 1Hz pulse wave is correct."""
    pulse_waveform = waveforms.Pulse(1, duty_cycle=0.75)

    sampled_pulse_waveform = pulse_waveform.sample_samples(8, 9)
    reference_pulse_samples = [1, 1, 1, 1, 1, 1, -1, -1, 1]
    assert sampled_pulse_waveform == pytest.approx(reference_pulse_samples)


def test_pulse_shape_2hz() -> None:
    """Ensure the shape of a 2Hz pulse wave is correct."""
    pulse_waveform = waveforms.Pulse(2, duty_cycle=0.75)

    sampled_pulse_waveform = pulse_waveform.sample_samples(8, 9)
    reference_pulse_samples = [1, 1, 1, -1, 1, 1, 1, -1, 1]
    assert sampled_pulse_waveform == pytest.approx(reference_pulse_samples)


def test_square_shape_1hz() -> None:
    """Ensure the shape of a 1Hz square wave is correct."""
    square_waveform = waveforms.Square(1)

    sampled_square_waveform = square_waveform.sample_samples(8, 9)
    reference_square_samples = [1, 1, 1, 1, -1, -1, -1, -1, 1]
    assert sampled_square_waveform == pytest.approx(reference_square_samples)


def test_square_shape_2hz() -> None:
    """Ensure the shape of a 2Hz square wave is correct."""
    square_waveform = waveforms.Square(2)

    sampled_square_waveform = square_waveform.sample_samples(8, 9)
    reference_square_samples = [1, 1, -1, -1, 1, 1, -1, -1, 1]
    assert sampled_square_waveform == pytest.approx(reference_square_samples)


def test_symmetric_triangle_shape_1hz() -> None:
    """Ensure the shape of a 1Hz symmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(1)

    sampled_triangle_waveform = triangle_waveform.sample_samples(8, 9)
    reference_triangle_samples = [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0]
    assert sampled_triangle_waveform == pytest.approx(reference_triangle_samples)


def test_symmetric_triangle_shape_2hz() -> None:
    """Ensure the shape of a 2Hz symmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(2)

    sampled_triangle_waveform = triangle_waveform.sample_samples(8, 9)
    reference_triangle_samples = [0, 1, 0, -1, 0, 1, 0, -1, 0]
    assert sampled_triangle_waveform == pytest.approx(reference_triangle_samples)


def test_asymmetric_triangle_shape_1hz() -> None:
    """Ensure the shape of a 1Hz asymmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(1, rising_part=0.75)

    sampled_triangle_waveform = triangle_waveform.sample_samples(8, 9)
    reference_triangle_samples = [0, 1 / 3, 2 / 3, 1, 0, -1, -2 / 3, -1 / 3, 0]
    assert sampled_triangle_waveform == pytest.approx(reference_triangle_samples)


def test_asymmetric_triangle_shape_2hz() -> None:
    """Ensure the shape of a 2Hz asymmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(2, rising_part=0.75)

    sampled_triangle_waveform = triangle_waveform.sample_samples(8, 9)
    reference_triangle_samples = [0, 2 / 3, 0, -2 / 3, 0, 2 / 3, 0, -2 / 3, 0]
    assert sampled_triangle_waveform == pytest.approx(reference_triangle_samples)


@pytest.mark.parametrize(
    "wave",
    [
        waveforms.Sine(440),
        waveforms.Sawtooth(440),
        waveforms.Pulse(440),
        waveforms.Pulse(440, duty_cycle=0.1),
        waveforms.Pulse(440, duty_cycle=0.9),
        waveforms.Square(440),
        waveforms.Triangle(440),
        waveforms.Triangle(440, rising_part=0.1),
        waveforms.Triangle(440, rising_part=0.9),
    ],
)
def test_wave_produces_correct_frequency(wave: waveforms.PeriodicWave) -> None:
    """Ensure periodic waves produce correct frequency.

    Args:
        wave: the periodic wave to test.
    """
    sampling_rate = 2000

    signal = wave.sample_seconds(sampling_rate, 1)

    peak_frequency = _get_peak_frequency(signal, sampling_rate)
    assert peak_frequency == wave.frequency


def test_phase_offset_does_not_affect_frequency() -> None:
    """Ensure that changing the phase offset of a periodic wave does not affect its frequency."""
    sampling_rate = 2000
    duration = 1
    wave = waveforms.Sine(440)

    signal_before = wave.sample_seconds(sampling_rate, duration)
    wave.phase = 0.37
    signal_after = wave.sample_seconds(sampling_rate, duration)

    peak_frequency_before = _get_peak_frequency(signal_before, sampling_rate)
    peak_frequency_after = _get_peak_frequency(signal_after, sampling_rate)
    assert peak_frequency_before == peak_frequency_after


def test_sampling_rate_does_not_affect_frequency() -> None:
    """Ensure that changing the sampling rate does not affect a periodic wave's frequency."""
    sampling_rate_1 = 2000
    sampling_rate_2 = 2530
    duration = 1
    wave = waveforms.Sine(440)

    signal1 = wave.sample_seconds(sampling_rate_1, duration)
    signal2 = wave.sample_seconds(sampling_rate_2, duration)

    peak_frequency_1 = _get_peak_frequency(signal1, sampling_rate_1)
    peak_frequency_2 = _get_peak_frequency(signal2, sampling_rate_2)
    assert peak_frequency_1 == peak_frequency_2


def test_changing_phase_offset_is_equivalent_to_changing_start_offset() -> None:
    """Ensure that the phase offset of a periodic phase is equivalent to sampling offset."""
    sampling_rate = 2000
    duration = 1
    wave = waveforms.Sine(1)

    signal_start_offset = wave.sample_seconds(sampling_rate, duration, 0.27)
    wave.phase = 0.27
    signal_phase_offset = wave.sample_seconds(sampling_rate, duration)

    assert signal_phase_offset == pytest.approx(signal_start_offset)


def test_sine_wave_with_half_phase_offset_equals_negative_sine_wave() -> None:
    """Ensure that a sine wave with 0.5 phase offset is equal to a negative sine wave."""
    sampling_rate = 2000
    duration = 1
    wave = waveforms.Sine(440)

    signal = wave.sample_seconds(sampling_rate, duration)
    wave.phase = 0.5
    offset_signal = wave.sample_seconds(sampling_rate, duration)

    assert offset_signal == pytest.approx(-signal)
