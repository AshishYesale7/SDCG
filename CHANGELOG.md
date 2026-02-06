# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-05

### Added
- Complete CGC (Casimir-Gravity Crossover) theoretical framework
- MCMC sampling with emcee for parameter estimation
- Multi-probe likelihood combining:
  - CMB (Planck 2018)
  - BAO (SDSS, BOSS, eBOSS)
  - Type Ia Supernovae (Pantheon+)
  - Growth rate measurements (fσ8)
  - Lyman-α forest power spectrum
- Chameleon screening implementation
- Environmental density dependence
- Publication-quality plotting tools
- Comprehensive documentation

### Changed
- Reorganized codebase structure:
  - All simulation code moved to `simulations/` directory
  - CGC module at `simulations/cgc/`
  - CLASS modifications at `simulations/class_cgc/`
- Fixed parameter values based on theory (Thesis v12):
  - n_g = 0.0125 (β₀²/4π²)
  - z_trans = 1.67 (cosmic dynamics)
  - α = 2.0 (Klein-Gordon)
  - ρ_thresh = 200 (virial theorem)

### Fixed
- Removed hardcoded file paths for portability
- Updated import paths for new structure
- Configuration auto-detects project root

## [0.9.0] - 2026-01-30

### Added
- Initial CGC physics implementation
- Basic MCMC framework
- Data loading utilities

### Known Issues
- Some hardcoded paths (fixed in 1.0.0)
- n_g was incorrectly fitted (now fixed to theory value)
