"""Tests for audio stream I/O."""

from typing import override
import dataclasses
import numpy as np
from glyphtune import io, signal, waveforms


@dataclasses.dataclass
class _FakeStreamIO:
    input: bytes = bytes()
    output: bytes = bytes()


_fake_stream_io = _FakeStreamIO()


class MockMonoStreamHandler(io.StreamHandler):
    """Fake stream handler that simply reads and writes from the global `_fake_stream_io` object."""

    @override
    def read(self, size: int) -> bytes:
        result = _fake_stream_io.input[: size * 4]
        _fake_stream_io.input = _fake_stream_io.input[size * 4 :]
        return result

    @override
    def write(self, data: bytes, size: int) -> None:
        _fake_stream_io.output += data

    @override
    def close(self) -> None: ...


class MockStereoStreamHandler(io.StreamHandler):
    """Fake stream handler that simply reads and writes from the global `_fake_stream_io` object."""

    @override
    def read(self, size: int) -> bytes:
        result = _fake_stream_io.input[: size * 8]
        _fake_stream_io.input = _fake_stream_io.input[size * 8 :]
        return result

    @override
    def write(self, data: bytes, size: int) -> None:
        _fake_stream_io.output += data

    @override
    def close(self) -> None: ...


def test_record_one_channel() -> None:
    """Ensure `record` with mono signal returns expected signal."""
    params = io.StreamParameters(1, 4, 4)
    data = [0, 0.25, 0.5, 0.75]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(1, params, MockMonoStreamHandler)

    reference_sig = signal.Signal([data])
    assert np.allclose(read_sig, reference_sig)


def test_record_two_channels() -> None:
    """Ensure `record` with stereo signal returns expected signal."""
    params = io.StreamParameters(2, 4, 4)
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(1, params, MockStereoStreamHandler)

    reference_sig = signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]])
    assert np.allclose(read_sig, reference_sig)


def test_record_one_channel_long_duration() -> None:
    """Ensure `record` with mono signal and long duration returns expected signal."""
    params = io.StreamParameters(1, 4, 4)
    data = [0, 0.25, 0.5, 0.75, 0.8, 0.95, 1, 0.7]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(2, params, MockMonoStreamHandler)

    reference_sig = signal.Signal([data])
    assert np.allclose(read_sig, reference_sig)


def test_record_one_channel_short_duration() -> None:
    """Ensure `record` with mono signal and short duration returns expected signal."""
    params = io.StreamParameters(1, 4, 1)
    data = [0, 0.25, 0.5, 0.75, 0.8, 0.95, 1, 0.7]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(0.5, params, MockMonoStreamHandler)

    reference_sig = signal.Signal([[0, 0.25]])
    assert np.allclose(read_sig, reference_sig)


def test_record_one_channel_small_buffer_size_divisible() -> None:
    """Ensure `record` with mono signal divisible by small buffer size returns expected signal."""
    params = io.StreamParameters(1, 4, 2)
    data = [0, 0.25, 0.5, 0.75]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(1, params, MockMonoStreamHandler)

    reference_sig = signal.Signal([[0, 0.25, 0.5, 0.75]])
    assert np.allclose(read_sig, reference_sig)


def test_record_one_channel_small_buffer_size_indivisible() -> None:
    """Ensure `record` with mono signal indivisible by small buffer size returns expected signal."""
    params = io.StreamParameters(1, 4, 3)
    data = [0, 0.25, 0.5, 0.75, 0.8, 0.95, 1, 0.7]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(1, params, MockMonoStreamHandler)

    reference_sig = signal.Signal([[0, 0.25, 0.5, 0.75, 0.8, 0.95]])
    assert np.allclose(read_sig, reference_sig)


def test_record_one_channel_large_buffer_size() -> None:
    """Ensure `record` with mono signal with large buffer size returns expected signal."""
    params = io.StreamParameters(1, 4, 5)
    data = [0, 0.25, 0.5, 0.75, 0.8, 0.95, 1, 0.7]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    read_sig = io.record(1, params, MockMonoStreamHandler)

    reference_sig = signal.Signal([[0, 0.25, 0.5, 0.75, 0.8]])
    assert np.allclose(read_sig, reference_sig)


def test_record_resample_two_channels() -> None:
    """Ensure `record_resample` with stereo signal returns expected waveform."""
    params = io.StreamParameters(2, 4, 4)
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    data_bytes = np.array(data).astype(np.float32).tobytes()
    _fake_stream_io.input = data_bytes

    resample_waveform = io.record_resample(1, params, MockStereoStreamHandler)

    reference_waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    assert resample_waveform.approx_equal(reference_waveform)


def test_play_one_channel() -> None:
    """Ensure `play` with mono signal has expected effect."""
    data = [0, 0.25, 0.5, 0.75]
    waveform = waveforms.ResampleWaveform(signal.Signal([data]), 4)
    params = io.StreamParameters(1, 4, 4)

    io.play(waveform, 1, params, 0, MockMonoStreamHandler)

    reference_bytes = np.array(data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_two_channels() -> None:
    """Ensure `play` with stereo signal has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.StreamParameters(2, 4, 4)
    io.play(waveform, 1, params, 0, MockStereoStreamHandler)

    reference_bytes = np.array(data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_one_channel_small_buffer_size_divisible() -> None:
    """Ensure `play` with mono signal divisible by small buffer size has expected effect."""
    data = [0, 0.25, 0.5, 0.75]
    waveform = waveforms.ResampleWaveform(signal.Signal([data]), 4)
    params = io.StreamParameters(1, 4, 2)

    io.play(waveform, 1, params, 0, MockMonoStreamHandler)

    reference_bytes = np.array(data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_one_channel_small_buffer_size_indivisible() -> None:
    """Ensure `play` with mono signal indivisible by small buffer size has expected effect."""
    data = [0, 0.25, 0.5, 0.75]
    waveform = waveforms.ResampleWaveform(signal.Signal([data]), 4)
    params = io.StreamParameters(1, 4, 3)
    duration = 1

    io.play(waveform, duration, params, 0, MockMonoStreamHandler)

    samples = duration * params.sampling_rate * params.channels
    padded_data = data + [0] * (params.buffer_size - samples % params.buffer_size)
    reference_bytes = np.array(padded_data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_one_channel_large_buffer_size() -> None:
    """Ensure `play` with mono signal and large buffer size has expected effect."""
    data = [0, 0.25, 0.5, 0.75]
    waveform = waveforms.ResampleWaveform(signal.Signal([data]), 4)
    params = io.StreamParameters(1, 4, 7)
    duration = 1

    io.play(waveform, duration, params, 0, MockMonoStreamHandler)

    samples = duration * params.sampling_rate * params.channels
    padded_data = data + [0] * (params.buffer_size - samples % params.buffer_size)
    reference_bytes = np.array(padded_data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_two_channels_longer_duration() -> None:
    """Ensure `play` with stereo signal and long duration has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.StreamParameters(2, 4, 2)

    io.play(waveform, 2, params, 0, MockStereoStreamHandler)

    padded_data = data + [0] * params.sampling_rate * params.channels
    reference_bytes = np.array(padded_data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_two_channels_shorter_duration() -> None:
    """Ensure `play` with stereo signal and short duration has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.StreamParameters(2, 4, 2)

    io.play(waveform, 0.5, params, 0, MockStereoStreamHandler)

    cut_data = data[: params.sampling_rate * params.channels // 2]
    reference_bytes = np.array(cut_data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_two_channels_with_offset() -> None:
    """Ensure `play` with stereo signal and offset has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.StreamParameters(2, 4, 2)

    io.play(waveform, 1, params, 0.5, MockStereoStreamHandler)

    half_samples = params.sampling_rate * params.channels // 2
    offset_data = data[half_samples:] + [0] * half_samples
    reference_bytes = np.array(offset_data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()


def test_play_two_channels_with_offset_and_shorter_duration() -> None:
    """Ensure `play` with stereo signal, offset, and short duration has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.StreamParameters(2, 4, 2)

    io.play(waveform, 0.5, params, 0.5, MockStereoStreamHandler)

    cut_offset_data = data[params.sampling_rate * params.channels // 2 :]
    reference_bytes = np.array(cut_offset_data).astype(np.float32).tobytes()
    assert _fake_stream_io.output == reference_bytes
    _fake_stream_io.output = bytes()
