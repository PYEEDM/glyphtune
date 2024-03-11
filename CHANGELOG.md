# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Output stream with an arbitrary number of channels (`glyphtune.output.Stream`).
- Effects module (including base `Effect` object).
- Stereo effects module (including `StereoPan`, `StereoLevels`, `StereoInterMix`, and `StereoDelay`).
- `glyphtune.arrays` module.

### Changed

- `Waveform.sample_arr` is now expected to be able to take as an argument an array of the shape `(channels, samples)` and return an array of the same shape.
- `Waveform.sample_seconds` and `Waveform.sample_samples` now accept a `channels` options and can sample multi-channel audio, returning arrays of the shape `(channels, samples)`.
- `Waveform.sample_seconds` and `Waveform.sample_samples` now sample stereo audio by default.
- `PhaseModulation` no longer accepts the `frequency_modulation` option and is no longer able to perform frequency modulation.
- `Square` objects no longer subclass `Pulse` and now subclass `PeriodicWave`.
- Moved `FloatArray` from `glyphtune` module to `glyphtune.arrays`.
- `PeriodicWave` now properly overrides `__eq__` and `__repr__`.
- Periodic wave representations now explicitly state `phase` keyword.
- `OperationWaveform` attribute `_operator_kwargs` is now `__operator_kwargs`. 

### Removed

- `glyphtune.output.MonoStream` class (replaced by `glyphtune.output.Stream`).
- Calculus module (`glyphtune.waveforms.calculus`, including `DerivativeWaveform` and `IntegralWaveform`).
- Frequency modulation (`glyphtune.waveforms.modulation.frequency_modulate`).

### Fixed

- Various bugfixes and improvements in equality checks.

## [0.1.0] 2024-03-03

### Added

- Initial version.

[0.1.0]: https://github.com/PYEEDM/glyphtune/releases/tag/0.1.0