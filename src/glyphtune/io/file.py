"""Audio output to file."""

from typing import override
import dataclasses
import pathlib
import wave
import numpy as np
from glyphtune import signal, waveforms


@dataclasses.dataclass
class FileParameters:
    """Parameters used in audio file I/O.

    Attributes:
        channels: number of channels.
        sample_width: the sample width in bytes.
        sampling_rate: the sampling rate in samples per second.
    """

    channels: int = 2
    sampling_rate: int = 44100
    sample_width: int = 4


class FileHandler:
    """Abstract class for handling reading and writing an audio file format.

    Attributes:
        path: the path of the file being handled.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path

    def read(self) -> tuple[FileParameters, bytes]:
        """Reads an audio file.

        Returns a tuple `(parameters, read_signal)` containing metadata and data read from the file.
        """
        raise NotImplementedError

    def write(self, parameters: FileParameters, data: bytes) -> None:
        """Writes an audio file.

        Args:
            parameters: file I/O parameters.
            data: audio bytes to write.
        """
        raise NotImplementedError


class WavHandler(FileHandler):
    """A file handler for uncompressed wav files in integer format."""

    @override
    def read(self) -> tuple[FileParameters, bytes]:
        wave_read = wave.Wave_read(str(self.path))
        parameters = FileParameters(
            wave_read.getnchannels(), wave_read.getnframes(), wave_read.getsampwidth()
        )
        read_bytes = wave_read.readframes(wave_read.getnframes())
        return parameters, read_bytes

    @override
    def write(self, parameters: FileParameters, data: bytes) -> None:
        wave_write = wave.Wave_write(str(self.path))
        wave_write.setnchannels(parameters.channels)
        wave_write.setsampwidth(parameters.sample_width)
        wave_write.setframerate(parameters.sampling_rate)
        wave_write.writeframes(data)
        wave_write.close()


_extension_to_handler = {".wav": WavHandler}


def _infer_handler(file_extension: str) -> type[FileHandler]:
    try:
        return _extension_to_handler[file_extension]
    except KeyError as exception:
        raise ValueError("Unsupported file extension") from exception


def read(
    path: pathlib.Path, handler_type: type[FileHandler] | None = None
) -> tuple[FileParameters, signal.Signal]:
    """Reads an audio file.

    Args:
        path: the path of the input file.
        handler_type: subclass of `FileHandler` that can handle the format of the input file.
            If None, an attempt will be made to find the right handler type from the file extension.

    Returns:
        A tuple `(parameters, read_signal)` containing metadata and data read from the file.
    """
    if handler_type is None:
        handler_type = _infer_handler(path.suffix)
    handler = handler_type(path)
    parameters, read_bytes = handler.read()
    type_code = f"i{parameters.sample_width}"
    type_max = np.iinfo(type_code).max
    read_array_flat = (
        np.frombuffer(read_bytes, dtype=type_code).astype(np.float32) / type_max
    )
    samples = len(read_array_flat) // parameters.channels
    read_signal = signal.Signal(
        read_array_flat.reshape((parameters.channels, samples), order="F")
    )
    return parameters, read_signal


def read_resample(
    path: pathlib.Path, handler_type: type[FileHandler] | None = None
) -> waveforms.ResampleWaveform:
    """Reads an aduio file into a resample waveform.

    Args:
        path: the path of the input file.
        handler_type: subclass of `FileHandler` that can handle the format of the input file.
            If None, an attempt will be made to find the right handler type from the file extension.

    Returns:
        A waveform that resamples the audio data read from file.
    """
    parameters, read_signal = read(path, handler_type)
    return waveforms.ResampleWaveform(read_signal, parameters.sampling_rate)


def write(
    waveform: waveforms.Waveform,
    path: pathlib.Path,
    duration: float,
    parameters: FileParameters,
    start_offset: float = 0,
    handler_type: type[FileHandler] | None = None,
) -> None:
    """Writes waveform to file.

    Args:
        waveform: the waveform to write.
        path: the path of the output file.
        duration: the duration of time to sample the waveform for output, in seconds.
        start_offset: the starting offset with which to sample the waveform for output, in seconds.
        parameters: file I/O parameters.
        handler_type: subclass of `FileHandler` that can handle the format of the input file.
            If None, an attempt will be made to find the right handler type from the file extension.
    """
    if handler_type is None:
        handler_type = _infer_handler(path.suffix)
    sig = waveform.sample_seconds(
        duration, parameters.sampling_rate, start_offset, parameters.channels
    )
    type_code = f"i{parameters.sample_width}"
    type_max = np.iinfo(type_code).max
    retyped_signal = np.asarray(sig * type_max).astype(type_code)
    signal_bytes = retyped_signal.tobytes("F")
    handler = handler_type(path)
    handler.write(parameters, signal_bytes)
