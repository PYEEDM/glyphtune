# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] 2024-03-21

### Added

- Effects module (including base `Effect` class).
- Stereo effects module (including `StereoPan`, `StereoLevels`, `StereoInterMix`, and `StereoDelay`).
- Resampling module (including `ResampleWaveform` class).
- Audio stream I/O with an arbitrary number of channels (`glyphtune.io.record`, `glyphtune.io.record_resample`, `glyphtune.io.play`, `glyphtune.io.StreamHandler`, `glyphtune.io.PyAudioHandler`, and `glyphtune.io.StreamParameters`).
- wav file I/O (`glyphtune.io.read`, `glyphtune.io.read_resample`, `glyphtune.io.write`, `glyphtune.io.FileHandler`, `glyphtune.io.WavHandler`, and `glyphtune.io.FileParameters`).
- `glyphtune.signal` module (including `Signal` class).

### Changed

- `Waveform.sample_arr` is renamed to `sample_time` and is now expected to be able to take as an argument a `Signal` object of the shape `(channels, samples)` and return a `Signal` object of the same shape.
- `Waveform.sample_seconds` and `Waveform.sample_samples` now accept a `channels` option and can sample multi-channel audio, returning `Signal` objects of the shape `(channels, samples)`.
- `Waveform.sample_seconds` and `Waveform.sample_samples` now sample stereo audio by default.
- `Waveform.sample_samples`'s `count` argument can now be set to `None` to sample one second.
- The order of arguments for `Waveform.sample_seconds` and `Waveform.sample_samples` changed (`duration` and `count` now come before `sampling_rate`).
- All arguments for `Waveform.sample_seconds` and `Waveform.sample_samples` now have default values (`duration=1`, `count=None`, and `sampling_rate=44100`)
- `PhaseModulation` no longer accepts the `frequency_modulation` option and is no longer able to perform frequency modulation.
- `PeriodicWave` now properly overrides `__eq__` and `__repr__`.
- Periodic wave representations now explicitly state `phase` keyword.
- `OperationWaveform` properties `operator`, `operands`, and `operator_kwargs` are now attributes and can therefore be freely accessed and set. 
- Various documentation improvements.

### Removed

- `glyphtune.output` module (replaced by `glyphtune.io`).
- Calculus module (`glyphtune.waveforms.calculus`, including `DerivativeWaveform` and `IntegralWaveform`).
- Frequency modulation (`glyphtune.waveforms.modulation.frequency_modulate`).
- `glyphtune.FloatArray` type alias.

### Fixed

- `Square` no longer subclasses `Pulse` and now subclasses `PeriodicWave`.
- Various bugfixes and improvements in equality checks.

## [0.1.0] 2024-03-03

### Added

- Initial version.

[0.1.0]: https://github.com/PYEEDM/glyphtune/releases/tag/0.1.0
[0.2.0]: https://github.com/PYEEDM/glyphtune/releases/tag/0.2.0