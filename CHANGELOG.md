# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `glyphtune.arrays` module.

### Changed

- `Square` objects no longer subclass `Pulse` and now subclass `PeriodicWave`.
- Moved `FloatArray` from `glyphtune` module to `glyphtune.arrays`.
- `PeriodicWave` now properly overrides `__eq__` and `__repr__`.
- Periodic wave representations now explicitly state `phase` keyword.
- `OperationWaveform` attribute `_operator_kwargs` is now `__operator_kwargs`. 

### Removed

- Calculus module (including DerivativeWaveform and IntegralWaveform)
- Frequency modulation

### Fixed

- Various bugfixes and improvements in equality checks.

## [0.1.0] 2024-03-03

### Added

- Initial version.

[0.1.0]: https://github.com/PYEEDM/glyphtune/releases/tag/0.1.0