"""Streaming audio output."""

import numpy as np
import pyaudio
from glyphtune import waveforms


class Stream:
    """Audio output stream."""

    def __init__(
        self, sampling_rate: int = 44100, buffer_size: int = 512, channels: int = 2
    ) -> None:
        """Initializes an audio stream.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            buffer_size: the size of chunks to be streamed in samples.
            channels: number of channels to output.
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.channels = channels

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of the output stream."""
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Sampling rate must be positive")
        self.__sampling_rate = value

    @property
    def buffer_size(self) -> int:
        """Buffer size of the output stream."""
        return self.__buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Buffer size must be positive")
        self.__buffer_size = value

    @property
    def channels(self) -> int:
        """Number of channels of the output stream."""
        return self.__channels

    @channels.setter
    def channels(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of channels must be positive")
        self.__channels = value

    def stream_waveform(self, waveform: waveforms.Waveform) -> None:
        """Stream a waveform until `SystemExit` or `KeyboardInterrupt` is raised.

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
                sampled_chunk = waveform.sample_samples(
                    self.sampling_rate,
                    self.buffer_size,
                    chunk_number * self.buffer_size,
                    self.channels,
                )
                sampled_chunk_bytes = sampled_chunk.astype(np.float32).tobytes("F")
                stream.write(sampled_chunk_bytes, self.buffer_size)
                chunk_number += 1
        except (SystemExit, KeyboardInterrupt):
            stream.stop_stream()
            stream.close()
            py_audio.terminate()
