"""Streaming audio output."""

from typing import override
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


class StreamHandler:
    """Abstract class for handling reading and writing to audio streams.

    Attributes:
        stream_parameters: audio I/O stream parameters.
    """

    def __init__(
        self, stream_parameters: StreamParameters = StreamParameters()
    ) -> None:
        self.stream_parameters = stream_parameters

    def read(self, size: int) -> bytes:
        """Returns audio data bytes from the stream.

        Args:
            size: the number of frames to read.
        """
        raise NotImplementedError

    def write(self, data: bytes, size: int) -> None:
        """Writes audio data bytes to the stream.

        Args:
            data: data to write.
            size: the number of frames to write.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Closes the stream."""
        raise NotImplementedError


class PyAudioHandler(StreamHandler):
    """A stream handler using PyAudio."""

    def __init__(
        self, stream_parameters: StreamParameters = StreamParameters()
    ) -> None:
        super().__init__(stream_parameters)
        self.__py_audio = pyaudio.PyAudio()
        self.__stream = self.__py_audio.open(
            self.stream_parameters.sampling_rate,
            self.stream_parameters.channels,
            pyaudio.paFloat32,
            input=True,
            output=True,
        )

    @override
    def read(self, size: int) -> bytes:
        return self.__stream.read(size)

    @override
    def write(self, data: bytes, size: int) -> None:
        self.__stream.write(data, size)

    @override
    def close(self) -> None:
        self.__stream.stop_stream()
        self.__stream.close()
        self.__py_audio.terminate()


def record(
    duration: float = np.inf,
    stream_parameters: StreamParameters = StreamParameters(1),
    handler_type: type[StreamHandler] = PyAudioHandler,
) -> signal.Signal:
    """Returns signal of recorded audio input.

    Args:
        duration: the duration of time to record input, in seconds.
            If set to infinity, recording will continue until interrupted.
            Note that the recording may last longer than the specified duration until
            the last buffer is over. This is more noticeable with a large `buffer_size`.
        stream_parameters: the stream parameters to use.
        handler_type: subclass of `StreamHandler` that can handle audio stream I/O.
    """
    handler = handler_type(stream_parameters)
    end_chunk = np.ceil(
        duration * stream_parameters.sampling_rate / stream_parameters.buffer_size
    )
    read_bytes = bytes()
    chunks = 0
    try:
        while chunks < end_chunk:
            read_bytes += handler.read(stream_parameters.buffer_size)
            chunks += 1
    except (KeyboardInterrupt, SystemExit) as exception:
        if isinstance(exception, SystemExit):
            handler.close()
            sys.exit()
    handler.close()
    read_array_flat = np.frombuffer(read_bytes, dtype=np.float32)
    read_signal = signal.Signal(
        read_array_flat.reshape(
            (stream_parameters.channels, stream_parameters.buffer_size * chunks),
            order="F",
        )
    )
    return read_signal


def record_resample(
    duration: float = np.inf,
    stream_parameters: StreamParameters = StreamParameters(1),
    handler_type: type[StreamHandler] = PyAudioHandler,
) -> waveforms.ResampleWaveform:
    """Returns a waveform that resamples the recorded audio input.

    Args:
        duration: the duration of time to record input, in seconds.
            If set to infinity, recording will continue until interrupted.
            Note that the recording may last longer than the specified duration until
            the last buffer is over. This is more noticeable with a large `buffer_size`.
        stream_parameters: the stream parameters to use.
        handler_type: subclass of `StreamHandler` that can handle audio stream I/O.
    """
    return waveforms.ResampleWaveform(
        record(duration, stream_parameters, handler_type),
        stream_parameters.sampling_rate,
    )


def play(
    waveform: waveforms.Waveform,
    duration: float = np.inf,
    stream_parameters: StreamParameters = StreamParameters(),
    start_offset: float = 0,
    handler_type: type[StreamHandler] = PyAudioHandler,
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
        handler_type: subclass of `StreamHandler` that can handle audio stream I/O.
    """
    handler = handler_type(stream_parameters)
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
            handler.write(chunk_signal_bytes, stream_parameters.buffer_size)
            chunk_number += 1
    except (SystemExit, KeyboardInterrupt):
        ...
    handler.close()
