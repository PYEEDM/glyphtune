"""Streaming audio output."""

import numpy as np
import pyaudio
from glyphtune import waveforms


class MonoStream:
    """Audio stream with mono output."""

    def __init__(
        self,
        sampling_rate: int = 44100,
        buffer_size: int = 512,
    ) -> None:
        """Initializes a mono audio stream.

        Args:
            sampling_rate: the sampling rate to use in samples per second.
            buffer_size: the size of chunks to be streamed in samples.
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size

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

    def stream_waveform(self, waveform: waveforms.Waveform) -> None:
        """Stream a waveform until `SystemExit` or `KeyboardInterrupt` is raised.

        Args:
            waveform: waveform to be streamed.
        """
        py_audio = pyaudio.PyAudio()
        stream = py_audio.open(self.__sampling_rate, 1, pyaudio.paFloat32, output=True)
        chunk_number = 0
        try:
            while True:
                sampled_chunk = waveform.sample_samples(
                    self.__sampling_rate,
                    self.buffer_size,
                    chunk_number * self.__buffer_size,
                )
                if sampled_chunk.dtype != np.float32:
                    sampled_chunk = sampled_chunk.astype(np.float32)
                chunk_bytes = sampled_chunk.tobytes()
                stream.write(chunk_bytes, self.__buffer_size)
                chunk_number += 1
        except (SystemExit, KeyboardInterrupt):
            stream.stop_stream()
            stream.close()
            py_audio.terminate()
