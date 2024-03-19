"""Tests for periodic waves."""

import numpy as np
import pytest
from glyphtune import signal, waveforms


def _get_peak_frequency(input_signal: signal.Signal, sampling_rate: int) -> float:
    """Return the peak frequency of the given signal's spectrum.

    Args:
        input_signal: mono signal to extract the peak frequency from.
        sampling_rate: the sampling rate of the signal in samples per second.
    """
    if not input_signal.is_mono:
        raise ValueError("Input signal is not mono")
    input_signal = input_signal.remove_dc_offset()
    spectrum = np.abs(np.fft.fft(np.squeeze(input_signal)))
    frequencies = np.fft.fftfreq(len(spectrum))
    peak_frequency = abs(frequencies[np.argmax(spectrum)] * sampling_rate)
    assert isinstance(peak_frequency, float)
    return peak_frequency


def test_sine_shape_1hz() -> None:
    """Ensure the shape of a 1Hz sine wave is correct."""
    sine_waveform = waveforms.Sine(1)

    sine_signal = sine_waveform.sample_samples(9, 8, channels=1)

    inv_sqrt_2 = 2**-0.5
    reference_sine_signal = signal.Signal(
        [[0, inv_sqrt_2, 1, inv_sqrt_2, 0, -inv_sqrt_2, -1, -inv_sqrt_2, 0]]
    )
    assert np.allclose(sine_signal, reference_sine_signal)


def test_sine_shape_2hz() -> None:
    """Ensure the shape of a 2Hz sine wave is correct."""
    sine_waveform = waveforms.Sine(2)

    sine_signal = sine_waveform.sample_samples(9, 8, channels=1)
    reference_sine_signal = signal.Signal([[0, 1, 0, -1, 0, 1, 0, -1, 0]])
    assert np.allclose(sine_signal, reference_sine_signal)


def test_sawtooth_shape_1hz() -> None:
    """Ensure the shape of a 1Hz sawtooth wave is correct."""
    sawtooth_waveform = waveforms.Sawtooth(1)

    sawtooth_signal = sawtooth_waveform.sample_samples(9, 8, channels=1)
    reference_sawtooth_signal = signal.Signal(
        [[0, 0.25, 0.5, 0.75, -1, -0.75, -0.5, -0.25, 0]]
    )
    assert np.allclose(sawtooth_signal, reference_sawtooth_signal)


def test_sawtooth_shape_2hz() -> None:
    """Ensure the shape of a 2Hz sawtooth wave is correct."""
    sawtooth_waveform = waveforms.Sawtooth(2)

    sawtooth_signal = sawtooth_waveform.sample_samples(9, 8, channels=1)
    reference_sawtooth_signal = signal.Signal([[0, 0.5, -1, -0.5, 0, 0.5, -1, -0.5, 0]])
    assert np.allclose(sawtooth_signal, reference_sawtooth_signal)


def test_pulse_shape_1hz() -> None:
    """Ensure the shape of a 1Hz pulse wave is correct."""
    pulse_waveform = waveforms.Pulse(1, duty_cycle=0.75)

    pulse_signal = pulse_waveform.sample_samples(9, 8, channels=1)
    reference_pulse_signal = signal.Signal([[1, 1, 1, 1, 1, 1, -1, -1, 1]])
    assert np.allclose(pulse_signal, reference_pulse_signal)


def test_pulse_shape_2hz() -> None:
    """Ensure the shape of a 2Hz pulse wave is correct."""
    pulse_waveform = waveforms.Pulse(2, duty_cycle=0.75)

    pulse_signal = pulse_waveform.sample_samples(9, 8, channels=1)
    reference_pulse_signal = signal.Signal([[1, 1, 1, -1, 1, 1, 1, -1, 1]])
    assert np.allclose(pulse_signal, reference_pulse_signal)


def test_square_shape_1hz() -> None:
    """Ensure the shape of a 1Hz square wave is correct."""
    square_waveform = waveforms.Square(1)

    square_signal = square_waveform.sample_samples(9, 8, channels=1)
    reference_square_signal = signal.Signal([[1, 1, 1, 1, -1, -1, -1, -1, 1]])
    assert np.allclose(square_signal, reference_square_signal)


def test_square_shape_2hz() -> None:
    """Ensure the shape of a 2Hz square wave is correct."""
    square_waveform = waveforms.Square(2)

    square_signal = square_waveform.sample_samples(9, 8, channels=1)
    reference_square_signal = signal.Signal([[1, 1, -1, -1, 1, 1, -1, -1, 1]])
    assert np.allclose(square_signal, reference_square_signal)


def test_symmetric_triangle_shape_1hz() -> None:
    """Ensure the shape of a 1Hz symmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(1)

    triangle_signal = triangle_waveform.sample_samples(9, 8, channels=1)
    reference_triangle_signal = signal.Signal([[0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0]])
    assert np.allclose(triangle_signal, reference_triangle_signal)


def test_symmetric_triangle_shape_2hz() -> None:
    """Ensure the shape of a 2Hz symmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(2)

    triangle_signal = triangle_waveform.sample_samples(9, 8, channels=1)
    reference_triangle_signal = signal.Signal([[0, 1, 0, -1, 0, 1, 0, -1, 0]])
    assert np.allclose(triangle_signal, reference_triangle_signal)


def test_asymmetric_triangle_shape_1hz() -> None:
    """Ensure the shape of a 1Hz asymmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(1, rising_part=0.75)

    triangle_signal = triangle_waveform.sample_samples(9, 8, channels=1)
    reference_triangle_signal = signal.Signal(
        [[0, 1 / 3, 2 / 3, 1, 0, -1, -2 / 3, -1 / 3, 0]]
    )
    assert np.allclose(triangle_signal, reference_triangle_signal)


def test_asymmetric_triangle_shape_2hz() -> None:
    """Ensure the shape of a 2Hz asymmetric triangle wave is correct."""
    triangle_waveform = waveforms.Triangle(2, rising_part=0.75)

    triangle_signal = triangle_waveform.sample_samples(9, 8, channels=1)
    reference_triangle_signal = signal.Signal(
        [[0, 2 / 3, 0, -2 / 3, 0, 2 / 3, 0, -2 / 3, 0]]
    )
    assert np.allclose(triangle_signal, reference_triangle_signal)


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

    wave_signal = wave.sample_seconds(1, sampling_rate, channels=1)

    peak_frequency = _get_peak_frequency(wave_signal, sampling_rate)
    assert peak_frequency == wave.frequency


def test_phase_offset_does_not_affect_frequency() -> None:
    """Ensure that changing the phase offset of a periodic wave does not affect its frequency."""
    sampling_rate = 2000
    duration = 1
    wave = waveforms.Sine(440)

    signal_before = wave.sample_seconds(duration, sampling_rate, channels=1)
    wave.phase = 0.37
    signal_after = wave.sample_seconds(duration, sampling_rate, channels=1)

    peak_frequency_before = _get_peak_frequency(signal_before, sampling_rate)
    peak_frequency_after = _get_peak_frequency(signal_after, sampling_rate)
    assert peak_frequency_before == peak_frequency_after


def test_sampling_rate_does_not_affect_frequency() -> None:
    """Ensure that changing the sampling rate does not affect a periodic wave's frequency."""
    sampling_rate_1 = 2000
    sampling_rate_2 = 2530
    duration = 1
    wave = waveforms.Sine(440)

    signal1 = wave.sample_seconds(duration, sampling_rate_1, channels=1)
    signal2 = wave.sample_seconds(duration, sampling_rate_2, channels=1)

    peak_frequency_1 = _get_peak_frequency(signal1, sampling_rate_1)
    peak_frequency_2 = _get_peak_frequency(signal2, sampling_rate_2)
    assert peak_frequency_1 == peak_frequency_2


def test_changing_phase_offset_is_equivalent_to_changing_start_offset() -> None:
    """Ensure that the phase offset of a periodic phase is equivalent to sampling offset."""
    sampling_rate = 2000
    duration = 1
    wave = waveforms.Sine(1)

    signal_start_offset = wave.sample_seconds(duration, sampling_rate, 0.27)
    wave.phase = 0.27
    signal_phase_offset = wave.sample_seconds(duration, sampling_rate)

    assert np.allclose(signal_start_offset, signal_phase_offset)


def test_sine_wave_with_half_phase_offset_equals_negative_sine_wave() -> None:
    """Ensure that a sine wave with 0.5 phase offset is equal to a negative sine wave."""
    sampling_rate = 2000
    duration = 1
    wave = waveforms.Sine(440)

    sine_signal = wave.sample_seconds(duration, sampling_rate)
    wave.phase = 0.5
    offset_signal = wave.sample_seconds(duration, sampling_rate)

    assert np.allclose(offset_signal, -sine_signal)
