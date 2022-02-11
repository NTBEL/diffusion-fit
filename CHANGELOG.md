# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.4.0] - 2022-02-08

### Added
- In DiffusionFitBase.fit function added new keyword argument `threshold_on` which sets how the peak and tail signal values for step 1 thresholding and termination of the fitting procedure computed. It has string options image, filter, line, and fit.
- New optional flag `-threshold-on` for the CLI script to set the `threshold_on` keyword argument to the fit function.
- Class variables `_threshold_on_options` and `_center_options` in ModeBase and some error checking to make sure the `center` and `threshold_on` keyword options  to `__init__` and `fit` are valid options.
- New fitting class `AnisotropicGaussianFit` for fitting cases where the diffusion coefficient is different along the major x and y axes, so the diffusion cloud is not isotropic.
- New optional flag `--anisotropic-gaussian` for the CLI script to use the `AnisotropicGaussianFit` class for fitting along with changes in the CLI script to adjust the DataFrame used to print values to the screen after all fitting is complete.
- In the CLI script an additional output of the DataFrame storing the diffusion coefficients and R-squared values for each image file to a csv file named diffusion_fitting/diffusion_fitting_summary.csv.
- In the CLI script and additional output of the input arguments to a text file diffusion_fitting/diffusionfit_commandline_args.txt.   
- Additional documentation on the programmatic and command line usage in the README.  
- New optional flag `--no-background` for the CLI script to set models not to try and compute or subtract background from the images.

### Changed
- In the export_to_csv function of DiffusionFitBase the addition of the linear fit column was changed to use the DataFrame `assign` function to get rid of the SettingWithCopyWarning. The column name was also changed from `Linear-Fit` to `LinearFit`.
- In the DiffusionFitBase.fit function the variables named signal were changed to peak to match the peak-to-tail thresholding naming during step 1 fitting.
- Updated the version numbers in the README.

### Fixed
- The issue with early termination due to lower than expected signal near the stimulation zone can be addressed using the newly added `threshold_on` option (`-threshold-on` from CLI). For example, using `threshold_on='fit'` will ensure smooth signal and tail regions consistent with the assumed intensity model.  
- In `measure.akaike_ic` the incorrect addition of the parametric term and max maximum_loglikelihood was fixed to be subtraction. This was causing an issue in the estimation of the time resolved diffusion coefficient.


## [0.3.0] - 2021-12-20

### Added
- Function to export fitting results to csv file (export_to_csv) in DiffusionFitBase.
- Abstract property function (fitting_parameters) to compile the fitting parameters from step 1 fitting into a DataFrame. This is used by export_to_csv function.
- __main__.py calls the export_to_csv function to save the fitting data to csv files.
- Time-resolved diffusion coefficient estimation (time_resolved_diffusion property function) and display function (display_time_resolved_dc). The plot is also part of the output files now when running from the command line (__main__.py).
- Added additional property functions to DiffusionFitBase:  fit_times, step1_rmse, step2_rsquared, effective_time.
- The command line run script (__main__.py) prints the Effective Time as part of Dstar_values DataFrame.
- New models module defining diffusion model functions to use when doing the fitting. Functions from this module are used by the fitting classes.
- New dependency on Numba and its use to improve performance of some numerical functions.
- New fitting class PointClarkFit for fitting fluorescent signal of receptor-based peptide sensors during peptide diffusion.
- Function in the DiffusionFitBase to write out the step 1 fits as an ImageJ compatible tif image trajectory.
- Funtionality to estimate the loss rate of the diffusing species.
- New optional input arguments for the command line version: -center, --time-resolved, --ignore-threshold, --write-tif, --loss-rate, --point-clark.
- Docstrings to functions in the models.py module.
- New pip install section in the README.

### Changed
- Changed the cmap used for step 1 experiment and 2D fit images from gray to viridis
- Replaced the ER goodness of fit metric with RSSE (Root Standard deviation of the Squared Error)
- Changed the way the thresholding is done after step 1 fitting. Now it terminates when mean(peak-region) <= mean(tail-region) + peak-to-tail * std(tail-region) and uses radial selections from the image instead of computing values from the Line-ROI.
- The required argument signal_to_noise in __main__.py arguments was changed to optional keyword argument -peak-to-tail with default value of 3. This is used in the new step 1 thresholding.
- In DiffusionFitBase class the member function `model` was changed to `intensity_model`, `linear_model` was changed to `diffusion_model`, and the `_fit_step1` and `fit_step2` functions were changed to `_fit_intensity` and `_fit_diffusion`, respectively.
- Updated the initial description, What's new in, License, and Documentation and Usage sections in the README.


## [0.2.0] - 2021-12-20

### Added
- New __main__.py so the package can be executed from the command line.
- Quality of fit metrics (root mean square error, RMSE, and error rate, ER) for the 2D image fitting which are printed in verbose output and on the plot generated by the display_image_fits function.
- pandas dependency.
- missing scikit-image dependency in the install_requires list of setup.py.
- optional tqdm dependency to display progress bar when running from the command line.

### Changed
- Changed diffusionfit.Gaussian class to diffusionfit.GaussianFit.
- Updated the region width used by the DiffusionFitBase.fit function when estimating the signal. It now uses 5 pixel widths instead of 3.
- Changed the way the noise is estimated. It now takes the average over the absolute values in the noise region instead of the standard deviation.
- Line ROI estimation via the DiffusionFitBase.line_average function uses the minimum size dimension (x or y) of the image now instead of just computing over the x-dimension. This also affects the noise estimation as the noise is estimated on the tails of the Line ROI.


## [0.1.0] - 2021-12-19

### Added
- Initial development version with 2D Gaussian fitting.

## [Unreleased] - yyyy-mm-dd

N/A

### Added

### Changed

### Fixed
