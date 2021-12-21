# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - yyyy-mm-dd

N/A

### Added

### Changed

### Fixed

## [0.2.0] - 2021-12-20

### Added
- New __main__.py so the package can be executed from the command line.
- Quality of fit metric for the 2D image fitting with a threshold to terminate the fitting procedure.

### Changed
- Changed diffusionfit.Gaussian class to diffusionfit.GaussianFit.
- Updated region widths used by the DiffusionFitBase.fit function when estimating the signal and noise. It now uses 5 pixel widths for the regions for both signal and noise instead of 3 and 10, respectively.
- Changed the way the noise is estimated. It now takes the average over the noise region instead of the standard deviation.
- Line ROI estimation via the DiffusionFitBase.line_average function uses the minimum size dimension (x or y) of the image now instead of just computing over the x-dimension. This also affects the noise estimation as the noise is estimated on the tails of the Line ROI.


## [0.1.0] - 2021-12-19

### Added
- Initial development version with 2D Gaussian fitting.
