"""Streaming audio output."""

import sys
import numpy as np
import pyaudio
from glyphtune import signal, waveforms


class IOStream:
    """Audio I/O stream."""

    def __init__(
        self, sampling_rate: int = 44100, buffer_size: int = 512, channels: int = 2
    ) -> None:
        """Initializes an audio I/O stream.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            buffer_size: the size of chunks to be streamed in samples.
            channels: number of channels.
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.channels = channels

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of the stream."""
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Sampling rate must be positive")
        self.__sampling_rate = value

    @property
    def buffer_size(self) -> int:
        """Buffer size of the stream."""
        return self.__buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Buffer size must be positive")
        self.__buffer_size = value

    @property
    def channels(self) -> int:
        """Number of channels of the stream."""
        return self.__channels

    @channels.setter
    def channels(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of channels must be positive")
        self.__channels = value

    def play(self, waveform: waveforms.Waveform) -> None:
        """Samples and streams a waveform's output until interrupted.

        Args:
            waveform: waveform to be streamed.
        """
        py_audio = pyaudio.PyAudio()
        stream = py_audio.open(
            self.sampling_rate, self.channels, pyaudio.paFloat32, output=True
        )
        chunk_number = 0
        try:
            while True:
                chunk_signal = waveform.sample_samples(
                    self.sampling_rate,
                    self.buffer_size,
                    chunk_number * self.buffer_size,
                    self.channels,
                )
                chunk_signal_bytes = chunk_signal.data.astype(np.float32).tobytes("F")
                stream.write(chunk_signal_bytes, self.buffer_size)
                chunk_number += 1
        except (SystemExit, KeyboardInterrupt):
            stream.stop_stream()
            stream.close()
            py_audio.terminate()

    def record(self) -> signal.Signal:
        """Returns signal of audio input that is recorded until interrupted."""
        py_audio = pyaudio.PyAudio()
        stream = py_audio.open(
            self.sampling_rate, self.channels, pyaudio.paFloat32, input=True
        )
        read_bytes = bytes()
        chunks = 0
        try:
            while True:
                read_bytes += stream.read(self.buffer_size)
                chunks += 1
        except (KeyboardInterrupt, SystemExit) as exception:
            stream.stop_stream()
            stream.close()
            py_audio.terminate()
            if isinstance(exception, SystemExit):
                sys.exit()
        read_array_flat = np.frombuffer(read_bytes, dtype=np.float32)
        read_signal = signal.Signal(
            read_array_flat.reshape((self.channels, self.buffer_size * chunks))
        )
        return read_signal
