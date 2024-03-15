"""Audio output to file."""

import dataclasses
import pathlib
import wave
import numpy as np
from glyphtune import waveforms


@dataclasses.dataclass
class WavParameters:
    """Parameters used in wav file output.

    Attributes:
        channels: number of channels.
        sample_width: the sample width in bytes.
        sampling_rate: the sampling rate in samples per second.
    """

    channels: int = 2
    sample_width: int = 4
    sampling_rate: int = 44100


def write_wav(
    waveform: waveforms.Waveform,
    path: pathlib.Path,
    duration: float,
    wav_parameters: WavParameters = WavParameters(),
    start_offset: float = 0,
) -> None:
    """Writes waveform to wav file.

    Args:
        waveform: the waveform to write.
        path: the path of the output file.
        duration: the duration of time to sample the waveform for output, in seconds.
        wav_parameters: wav file output parameters.
        start_offset: the starting offset with which to sample the waveform for output, in seconds.
    """
    signal = waveform.sample_seconds(
        wav_parameters.sampling_rate, duration, start_offset, wav_parameters.channels
    )
    type_code = f"i{wav_parameters.sample_width}"
    type_max = np.iinfo(type_code).max
    retyped_signal = (signal * type_max).array.astype(type_code)
    signal_bytes = retyped_signal.tobytes("F")
    wave_write = wave.Wave_write(str(path))
    wave_write.setnchannels(wav_parameters.channels)
    wave_write.setsampwidth(wav_parameters.sample_width)
    wave_write.setframerate(wav_parameters.sampling_rate)
    wave_write.writeframes(signal_bytes)
    wave_write.close()
