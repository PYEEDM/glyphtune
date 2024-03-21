"""Tests for audio file I/O."""

from typing import override
import dataclasses
import numpy as np
from glyphtune import io, signal, waveforms


@dataclasses.dataclass
class _FakeFileIO:
    input: tuple[io.FileParameters, bytes] = (io.FileParameters(), bytes())
    output: bytes = bytes()


_fake_file_io = _FakeFileIO()


class MockFileHandler(io.FileHandler):
    """Fake file handler that simply reads and writes from the global `_fake_file_io` object."""

    @override
    def read(self) -> tuple[io.FileParameters, bytes]:
        return _fake_file_io.input

    @override
    def write(self, parameters: io.FileParameters, data: bytes) -> None:
        _fake_file_io.output = data


def test_read_one_channel_int32() -> None:
    """Ensure `read` with mono signal and int32 returns expected signal."""
    params = io.FileParameters(1, 4, 4)
    data = [0, 0.25, 0.5, 0.75]
    data_bytes = (np.array(data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    _fake_file_io.input = (params, data_bytes)

    read_params, read_sig = io.read("UNUSED_MOCK", MockFileHandler)

    reference_sig = signal.Signal([data])
    assert read_params == params and np.allclose(read_sig, reference_sig)


def test_read_two_channels_int16() -> None:
    """Ensure `read` with stereo signal and int16 returns expected signal."""
    params = io.FileParameters(2, 4, 2)
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    data_bytes = (np.array(data) * np.iinfo(np.int16).max).astype(np.int16).tobytes()
    _fake_file_io.input = (params, data_bytes)

    read_params, read_sig = io.read("UNUSED_MOCK", MockFileHandler)

    reference_sig = signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]])
    assert read_params == params and np.allclose(read_sig, reference_sig, atol=10e-5)


def test_read_resample_one_channel_int32() -> None:
    """Ensure `read_resample` with mono signal and int32 returns expected waveform."""
    params = io.FileParameters(1, 4, 4)
    data = [0, 0.25, 0.5, 0.75]
    data_bytes = (np.array(data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    _fake_file_io.input = (params, data_bytes)

    resample_waveform = io.read_resample("UNUSED_MOCK", MockFileHandler)

    reference_waveform = waveforms.ResampleWaveform(signal.Signal([data]), 4)
    assert resample_waveform.approx_equal(reference_waveform)


def test_write_two_channels_int16() -> None:
    """Ensure `write` with stereo signal and int16 has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.FileParameters(2, 4, 2)

    io.write(waveform, "UNUSED_MOCK", 1, params, 0, MockFileHandler)

    reference_bytes = (
        (np.array(data) * np.iinfo(np.int16).max).astype(np.int16).tobytes()
    )
    assert _fake_file_io.output == reference_bytes


def test_write_one_channel_int32() -> None:
    """Ensure `write` with mono signal and int32 has expected effect."""
    data = [0, 0.25, 0.5, 0.75]
    waveform = waveforms.ResampleWaveform(signal.Signal([[0, 0.25, 0.5, 0.75]]), 4)
    params = io.FileParameters(1, 4, 4)

    io.write(waveform, "UNUSED_MOCK", 1, params, 0, MockFileHandler)

    reference_bytes = (
        (np.array(data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    )
    assert _fake_file_io.output == reference_bytes


def test_write_two_channels_int32_longer_duration() -> None:
    """Ensure `write` with stereo signal, int32, and long duration has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.FileParameters(2, 4, 4)

    io.write(waveform, "UNUSED_MOCK", 2, params, 0, MockFileHandler)

    padded_data = data + [0] * params.sampling_rate * params.channels
    reference_bytes = (
        (np.array(padded_data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    )
    assert _fake_file_io.output == reference_bytes


def test_write_two_channels_int32_shorter_duration() -> None:
    """Ensure `write` with stereo signal, int32, and short duration has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.FileParameters(2, 4, 4)

    io.write(waveform, "UNUSED_MOCK", 0.5, params, 0, MockFileHandler)

    cut_data = data[: params.sampling_rate * params.channels // 2]
    reference_bytes = (
        (np.array(cut_data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    )
    assert _fake_file_io.output == reference_bytes


def test_write_two_channels_int32_with_offset() -> None:
    """Ensure `write` with stereo signal, int32, and offset has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.FileParameters(2, 4, 4)

    io.write(waveform, "UNUSED_MOCK", 1, params, 0.5, MockFileHandler)

    half_samples = params.sampling_rate * params.channels // 2
    offset_data = data[half_samples:] + [0] * half_samples
    reference_bytes = (
        (np.array(offset_data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    )
    assert _fake_file_io.output == reference_bytes


def test_write_two_channels_int32_with_offset_and_shorter_duration() -> None:
    """Ensure `write` with stereo signal, int32, offset, and short duration has expected effect."""
    data = [0, 0, 0.1, -0.1, 0.2, 0.4, 0.5, -0.3]
    waveform = waveforms.ResampleWaveform(
        signal.Signal([[0, 0.1, 0.2, 0.5], [0, -0.1, 0.4, -0.3]]), 4
    )
    params = io.FileParameters(2, 4, 4)

    io.write(waveform, "UNUSED_MOCK", 0.5, params, 0.5, MockFileHandler)

    cut_offset_data = data[params.sampling_rate * params.channels // 2 :]
    reference_bytes = (
        (np.array(cut_offset_data) * np.iinfo(np.int32).max).astype(np.int32).tobytes()
    )
    assert _fake_file_io.output == reference_bytes
