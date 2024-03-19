"""Audio output to file."""

import dataclasses
import pathlib
import wave
import numpy as np
from glyphtune import signal, waveforms


@dataclasses.dataclass
class WavParameters:
    """Parameters used in wav file I/O.

    Attributes:
        channels: number of channels.
        sample_width: the sample width in bytes.
        sampling_rate: the sampling rate in samples per second.
    """

    channels: int = 2
    sample_width: int = 4
    sampling_rate: int = 44100


def read_wav(path: pathlib.Path) -> tuple[WavParameters, signal.Signal]:
    """Reads a wav file.

    Args:
        path: the path of the input file.

    Returns:
        A tuple `(wav_parameters, read_signal)` containing the metadata and data read from the file.
    """
    wave_read = wave.Wave_read(str(path))
    wav_parameters = WavParameters(
        wave_read.getnchannels(),
        wave_read.getsampwidth(),
        wave_read.getframerate(),
    )

    length = wave_read.getnframes()
    read_bytes = wave_read.readframes(length)
    type_code = f"i{wav_parameters.sample_width}"
    type_max = np.iinfo(type_code).max
    read_array_flat = (
        np.frombuffer(read_bytes, dtype=type_code).astype(np.float32) / type_max
    )
    read_signal = signal.Signal(
        read_array_flat.reshape((wav_parameters.channels, length))
    )
    return wav_parameters, read_signal


def write_wav(
    waveform: waveforms.Waveform,
    path: pathlib.Path,
    duration: float,
    start_offset: float = 0,
    wav_parameters: WavParameters = WavParameters(),
) -> None:
    """Writes waveform to wav file.

    Args:
        waveform: the waveform to write.
        path: the path of the output file.
        duration: the duration of time to sample the waveform for output, in seconds.
        start_offset: the starting offset with which to sample the waveform for output, in seconds.
        wav_parameters: wav file output parameters.
    """
    sig = waveform.sample_seconds(
        duration, wav_parameters.sampling_rate, start_offset, wav_parameters.channels
    )
    type_code = f"i{wav_parameters.sample_width}"
    type_max = np.iinfo(type_code).max
    retyped_signal = np.asarray(sig * type_max).astype(type_code)
    signal_bytes = retyped_signal.tobytes("F")
    wave_write = wave.Wave_write(str(path))
    wave_write.setnchannels(wav_parameters.channels)
    wave_write.setsampwidth(wav_parameters.sample_width)
    wave_write.setframerate(wav_parameters.sampling_rate)
    wave_write.writeframes(signal_bytes)
    wave_write.close()
