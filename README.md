# diffusion-fit



**diffusion-fit** is a python package that defines models to fit the 2D fluorescence intensity distribution in a time lapse series of fluorescence microscope images and extract estimates of a diffusion coeffient using a point-source paradigm.  

The underlying point-source paradigm and two-step fitting procedure employed by the models is adapted from the methods used to analyze integrative optical imaging data ([Nicholson and Tao 1993](https://doi.org/10.1016/S0006-3495(93)81324-9) and [Hrabe and Hrabetova 2019](https://doi.org/10.1016/j.bpj.2019.08.031)). **diffusion-fit** therefore provides a model for fitting 2D imaging data for fluorescent dyes released from a point-like source as in integrative optical imaging ([Nicholson and Tao 1993](https://doi.org/10.1016/S0006-3495(93)81324-9)) or nanovesicle photorelease ([Xiong et al. 2021](https://doi.org/10.1101/2021.09.10.459853)) experiments.

Additionally, we have extended the framework to provide a model for fitting data from receptor-based fluorescent peptide sensors, such as dLight ([Patriarchi et al. 2018](https://dx.doi.org/10.1126/science.aat4422)) and kLight ([Abraham et al. 2021](https://doi.org/10.1038/s41386-021-01168-2)), when the target peptides are released from a point-like source as in nanovesicle photorelease ([Xiong et al. 2021](https://doi.org/10.1101/2021.09.10.459853)) experiments.
**diffusion-fit** also further extends the point-source paradigm to allow for additional fitting and estimation of first order loss rate constants from the 2D imaging data, which are particularly relevant for peptide volume transmission in the brain ([Xiong et al. 2021](https://doi.org/10.1101/2021.09.10.459853)).  

### What's new in

#### version 0.6.0
 * The command line interface now accepts float values for the `-peak-to-tail` input option instead of just integers.

See the [CHANGELOG](CHANGELOG.md) for additional details.  

## Table of Contents

 1. [Install](#install)
     1. [Dependencies](#dependencies)
     2. [pip install](#pip-install)
     3. [Manual install](#manual-install)
     4. [Recommended additional software](#recommended-additional-software)
 2. [License](#license)
 3. [Change Log](#change-log)
 4. [Documentation and Usage](#documentation-and-usage)
     1. [Quick Overview](#quick-overview)
     2. [Programmatic use](#programmatic-use)
     3. [Command line Use](#command-line-use)
 5. [Contact](#contact)
 6. [Citing](#citing)  

------

# Install

| **! Note** |
| :--- |
|  diffusion-fit is still in version zero development so new versions may not be backwards compatible. |

**diffusion-fit** installs as the `diffusionfit` Python package. It is tested with Python 3.9.

### Dependencies
Note that `diffusion-fit` has the following core dependencies:
   * [NumPy](http://www.numpy.org/)
   * [SciPy](https://www.scipy.org/)
   * [scikit-image](https://scikit-image.org/)
   * [pandas](https://pandas.pydata.org/)
   * [Matplotlib](https://matplotlib.org/)
   * [seaborn](https://seaborn.pydata.org/)
   * [Numba](https://numba.pydata.org/)
   * [streamlit](https://streamlit.io/)
   * [plotly](https://plotly.com/)

### pip install
You can install `diffusionfit` version 0.6.0 with `pip` sourced from the GitHub repo:

##### with git installed:
Fresh install:
```
pip install git+https://github.com/NTBEL/diffusion-fit@v0.7.0
```
Or to upgrade from an older version:
```
pip install --upgrade git+https://github.com/NTBEL/diffusion-fit@v0.7.0
```
##### without git installed:
Fresh install:
```
pip install https://github.com/NTBEL/diffusion-fit/archive/refs/tags/v0.7.0.zip
```
Or to upgrade from an older version:
```
pip install --upgrade https://github.com/NTBEL/diffusion-fit/archive/refs/tags/v0.7.0.zip
```
### Manual install
First, download the repository. Then from the `diffusion-fit` folder/directory run
```
pip install .
```

### Recommended additional software

The following software is not required for the basic operation of **diffusonfit**, but provide extra capabilities and features when installed.

#### tqdm
Command line runs will display a progress bar that tracks the analysis of the set of image files when the [tqdm](https://github.com/tqdm/tqdm) package installed.  

------

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details

------

# Change Log

See: [CHANGELOG](CHANGELOG.md)

------

# Documentation and Usage

### Quick Overview
Currently, `diffusionfit` defines the **GaussianFit**, **PointClarkFit**, and **AnisotropicGaussianFit** classes, which define models to fit the 2D fluorescence intensity distribution and estimate the diffusion coefficient of
molecules released from a point-like source:
  * **GaussianFit** assumes the fluorescence signal comes directly from the diffusing species and that the diffusion cloud has an isotropic Gaussian distribution that expands in width over time as in the point source diffusion model.
  * **PointClarkFit** assumes the fluorescent signal comes from the diffusing species binding to a fluorescent receptor-based sensor such as the dLight or kLight peptide sensors. The diffusion is still assumed to follow the isotropic point source model but the resulting fluorescent signal follows the Clark equation for receptor-response with the diffusing molecule as the ligand and the fluorescent sensor as the receptor.
  * **AnisotropicGaussianFit** is also based on the point source diffusion model and assumes the fluorescence signal comes directly from the diffusing species, but allows the diffusion cloud to have an anisotropic Gaussian distribution that expands in width at different rates along the x and y dimensions over time. This model decouples diffusion in the x and y directions allowing the two dimensions to have different diffusion coefficients.

In all cases, the 2D fluorescence distribution of each image in the time lapse of fluorescence images is fitted with the 2D intensity model and then the diffusion coefficient is extracted in a second linear fitting step. This two-step fitting procedure was adapted from the methods used to analyze integrative optical imaging data as described in Nicholson and Tao 1993 [doi: 10.1016/S0006-3495(93)81324-9](https://doi.org/10.1016/S0006-3495(93)81324-9) and Hrabe and Hrabetova 2019 [doi: 10.1016/j.bpj.2019.08.031](https://doi.org/10.1016/j.bpj.2019.08.031).

`diffusionfit` can be used programatically, from the command line, or via a locally streamlit app. The command line interface simplifies the process for quickly initiating analysis of multiple image files (tiff files), and it automatically generates plot images of the fits and exports the fitting parameters to a csv file for further analysis for each image file (dumped into a new output fold named diffusion_fitting). The streamlit app only allows for working with one file at a time but the graphical interface makes it much easier to interact with the fitting procedure. However, programmatic use offers more flexibility for generating customized fitting and analysis workflows.   

### Programmatic use
The fitting models can be directly imported from `diffusionfit`:
```python
from diffusonfit import GaussianFit, PointClarkFit, AnisotropicGaussianFit
```
A fitting model can be initialized with the time lapse image file (in tiff format) and parameters describing the image as in this example:
```python
# Units:
# timestep (s)
# pixel_width (micron)
# stimulation_radius (micron)
gfit = GaussianFit('images.tif',
                    stimulation_frame=50, timestep=0.6,
                    pixel_width=0.99, stimulation_radius=30,
                    center='image', subtract_background=True)                   
```
After initialization, the fitting process is performed by calling the `fit` function which returns an estimate of the effective diffusion coefficient:
```python
Dstar = gfit.fit(verbose=True, step1_threshold=3)
```
The `display_image_fits` and `display_linear_fit` functions can be used to generate and output plots corresponding to each of the two fitting steps:
```python
# Step 1
gfit.display_image_fits(saveas='step1_fits.png')
# Step 2
gfit.display_linear_fit(saveas='step2_linfit.png')
```

### Command line use

When called from the command line diffusionfit will analyze and fit all time lapse image files in the current directory/folder using the same fitting settings. Image files are expected to be in the multi-frame tiff format. If different image files require different setting they can be moved into separate folders and diffusion fit can be called with the appropriate input arguments for each case.

#### Input arguments
There are four required positional arguments:
```
python -m diffusionfit [timestep] [pixel_size] [stim_frame] [d_stim]
```
which are:
  * **timestep** - the time interval between the images of the time lapse given in seconds.
  * **pixel_size** - the width of each image pixel in microns.
  * **stim_frame** - the frame where release of the diffusing species (e.g., by photostimulation) occurs in the time lapse. Images before this point are used to calculate the background for background subtraction.
  * **d_stim** - the diameter in microns of the stimulation zone when a tornado scan is used for photorelease. This zone is excluded from fitting due to the possibility of thermal damage or other effects during photorelease. If you don't need or want to account for the stimulation zone (e.g., analyzing IOI data) then you can set the value to `0`.

Additionally, there a many optional input flags that can be used to tune the fitting and its outputs. They are:
  * `-peak-to-tail [peak_to_tail]` : Set the peak/tail threshold during step 1 fitting for terminating the fitting analysis.
  * `-center [center]` : Set how the center of the diffusion cloud is determined. Options are: image - (default), use the center pixel location of the image. intensity - centroid of intensity after stimulation. y,x - a specific pixel location.
  * `--ignore-threshold`  :  The fitting model will ignore thresholding used to determine when during the time lapse to terminate the fitting early due to low signal in an image.
  * `-end-frame [end_frame]` : You can specify the maximum frame of the time lapse to include in the analysis. It should be larger than stim_frame. This can used in combination with the --ignore-threshold option to define an exact stopping point for the fitting in cases such as when you want to fit a pre-defined time interval or you have already pre-analyzed the images to determine when to terminate the fitting analysis.
  * `--write-tif` : An ImageJ compatible tiff time lapse image file will be written out with the 2D fit to the intensity of each image included in the analysis. This can be used for diagnostic purposes to visually inspect how the fits compare to the original images.
  * `--time-resolved` : The fitter will compute an estimate of time-resolved diffusion coefficient and output a corresponding plot. This is adapted from the TR-IOI approach described in Hrabe and Hrabetova 2019 [doi: 10.1016/j.bpj.2019.08.031](https://doi.org/10.1016/j.bpj.2019.08.031).
  * `--loss-rate` : The fitter will compute and output an estimate of the loss rate for the diffusing species from the decay of the maximum intensity fitting parameter using a model for its time course that is derived from the point source model.
  * `--point-clark` : Fit the intensity using the Point-Clark model (PointClarkFit). The Point-Clark model is derived from a combination of the point source diffusion model and the Clark equation for receptor-response (Hill equation with Hill coefficient of 1). This model should provide a more accurate estimate of peptide diffusion where the intensity comes from a fluorescent receptor-based sensor.
  * `-threshold-on [threshold_on]` : This option can be used to set how the peak and tail tail values for thresholding the fitting are computed. Options are: image - (default), compute on the (background subtracted) image. filter - compute on a Gaussian filtered version of the image. line - compute on the line ROI taken along the minimum image dimension. fit - compute on the fit of the image to the intensity model.
  * `-threshold-noise [threshold_noise]` : This option can be used to set how the noise in the tail region is determined for thresholding the fitting. Options are: std_dev - (default), use standard deviation of signal in the tail region. std_error - use the standard error in the tail region determined as std-dev/sqrt(N_values).
  * `--anisotropic-gaussian` : Use the Anisotropic Gaussian model (AnisotropicGaussianFit) to fit the diffusion intensity for cases where diffusion along the x and y dimensions is different.
  * `--no-background` : Don't compute or subtract any background from the images when fitting the intensity. This option can be used in cases where images are already pre-processed to remove the background flurorescence or when the input has been converted to the relative difference (dF/F) in flurorescence intensity (relevant for peptide sensors).     

The usage amd full set of command line options can be accessed at the command line by using the `--help` flag:
```
python -m diffusionfit --help
```

#### Outputs
The command line call to diffusionfit will generate a new folder `diffusion_fitting` where it will write output files.

Two summary files are written out:
  * **diffusionfit_commandline_args.txt** : A text file that stores the input parameters passed to diffusionfit. This can be used to recreate the fitting again later with the same settings or for archival purposes.
  * **diffusion_fitting_summary.csv** : csv file of summary values for the fitting. It contains the image file name and corresponding mean value of root mean squared errors from step 1 fitting and its standard deviation, the effective time, the estimated diffusion coefficient(s), and the R-squared from the linear fitting in step 2. It will also contain the loss rate if the optional `--loss-rate` flag is used.

The following output files are generated for each image file ([image_file_name].tif):  
  * **[image_file_name]_step_1_fits.csv** : csv file with the fitting and diagnostic parameters from the intensity model as fitted to each image in the time lapse.
  * **[image_file_name_step_2_fits.csv]** : csv file with the fitting parameters and linear fit used in step 2.
  * **[image_file_name]_step1.png** : An image file with plots comparing the original image to its 2D fit, a line ROI comparison, and a Ring ROI comparison. To keep the size of the image somewhat reasonable only a subset of the time lapse are shown.
  * **[image_file_name]_step2.png** : An image file with the plot(s) of the step 2 linear fitting.

#### Example
diffusionfit can be used from the command line as in the following example:
```
python -m diffusionfit 0.25 2.486 50 75 --loss-rate
```

### streamlit app use
The diffusionfit streamlit app can be launched from the command line as follows:
```
python -m diffusionfit.run_app
```
This should launch the streamlit app locally in your browser. From there you can upload data and run the fitting. 
------

# Contact

Please open a [GitHub Issue](https://github.com/NTBEL/diffusion-fit/issues) to
report any problems/bugs or make any comments, suggestions, or feature requests.

------

# Citing
If this package is useful in your research, please cite this GitHub repo: https://github.com/NTBEL/diffusion-fit
