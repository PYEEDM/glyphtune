"""Streaming audio output."""

import dataclasses
import sys
import numpy as np
import pyaudio
from glyphtune import signal, waveforms


@dataclasses.dataclass
class StreamParameters:
    """Parameters used in audio stream I/O.

    Attributes:
        channels: number of channels.
        sampling_rate: the sampling rate in samples per second.
        buffer_size: the size of chunks to be streamed in samples.
    """

    channels: int = 2
    sampling_rate: int = 44100
    buffer_size: int = 512


def play(
    waveform: waveforms.Waveform,
    duration: float = np.inf,
    start_offset: float = 0,
    stream_parameters: StreamParameters = StreamParameters(),
) -> None:
    """Samples and streams a waveform's output.

    Args:
        waveform: waveform to be streamed.
        duration: the duration of time to sample the waveform for output, in seconds.
            If set to infinity, playback will continue until interrupted.
            Note that the playback may last longer than the specified duration until
            the last buffer is over. This is more noticeable with a large `buffer_size`.
        start_offset: the starting offset with which to sample the waveform for output,
            in seconds.
        stream_parameters: the stream parameters to use.
    """
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(
        stream_parameters.sampling_rate,
        stream_parameters.channels,
        pyaudio.paFloat32,
        output=True,
    )
    start_sample = int(start_offset * stream_parameters.sampling_rate)
    end_chunk = np.ceil(
        duration * stream_parameters.sampling_rate / stream_parameters.buffer_size
    )
    chunk_number = 0
    try:
        while chunk_number < end_chunk:
            chunk_signal = waveform.sample_samples(
                stream_parameters.buffer_size,
                stream_parameters.sampling_rate,
                start_sample + chunk_number * stream_parameters.buffer_size,
                stream_parameters.channels,
            )
            chunk_signal_bytes = chunk_signal.data.astype(np.float32).tobytes("F")
            stream.write(chunk_signal_bytes, stream_parameters.buffer_size)
            chunk_number += 1
    except (SystemExit, KeyboardInterrupt):
        stream.stop_stream()
        stream.close()
        py_audio.terminate()


def record(
    duration: float = np.inf, stream_parameters: StreamParameters = StreamParameters(1)
) -> signal.Signal:
    """Returns signal of recorded audio input.

    Args:
        duration: the duration of time to record input, in seconds.
            If set to infinity, recording will continue until interrupted.
            Note that the recording may last longer than the specified duration until
            the last buffer is over. This is more noticeable with a large `buffer_size`.
        stream_parameters: the stream parameters to use.
    """
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(
        stream_parameters.sampling_rate,
        stream_parameters.channels,
        pyaudio.paFloat32,
        input=True,
    )
    end_chunk = np.ceil(
        duration * stream_parameters.sampling_rate / stream_parameters.buffer_size
    )
    read_bytes = bytes()
    chunks = 0
    try:
        while chunks < end_chunk:
            read_bytes += stream.read(stream_parameters.buffer_size)
            chunks += 1
    except (KeyboardInterrupt, SystemExit) as exception:
        stream.stop_stream()
        stream.close()
        py_audio.terminate()
        if isinstance(exception, SystemExit):
            sys.exit()
    read_array_flat = np.frombuffer(read_bytes, dtype=np.float32)
    read_signal = signal.Signal(
        read_array_flat.reshape(
            (stream_parameters.channels, stream_parameters.buffer_size * chunks),
            order="F",
        )
    )
    return read_signal


def record_resample(
    duration: float = np.inf, stream_parameters: StreamParameters = StreamParameters(1)
) -> waveforms.ResampleWaveform:
    """Returs a resample waveform of recorded audio input.

    Args:
        duration: the duration of time to record input, in seconds.
            If set to infinity, recording will continue until interrupted.
            Note that the recording may last longer than the specified duration until
            the last buffer is over. This is more noticeable with a large `buffer_size`.
        stream_parameters: the stream parameters to use.
    """
    return waveforms.ResampleWaveform(
        record(duration, stream_parameters), stream_parameters.sampling_rate
    )
