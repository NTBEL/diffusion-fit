# diffusion-fit



**diffusion-fit** is a python package for extract estimates of dye/peptide diffusion coefficients and loss rates from a time-sequence of fluorescence images.

### What's new in

#### version 0.2.0
 * New command line interface so that diffusionfit can be run quickly from the command line.
 * Optional progress bar to monitor analysis of image files when running diffusionfit from the command line. The progress bar is only displayed if [tqdm](https://github.com/tqdm/tqdm) is installed  
 * Updated estimation of noise when signal/noise thresholding images during analysis.
 * Automatic detection of the smallest dimension when the image is not square. Line ROIs are accordingly only estimated along this dimension and images are only fit at distances less than that limited distance.
 * Quality of fit metrics are estimated for the image fitting and displayed on the corresponding plots.

See the [CHANGELOG](CHANGELOG.md) for additional details.  

## Table of Contents

 1. [Install](#install)
     1. [Dependencies](#dependencies)
     2. [Manual install](#manual-install)
     3. [Recommended additional software](#recommended-additional-software)
 2. [License](#license)
 3. [Change Log](#change-log)
 4. [Documentation and Usage](#documentation-and-usage)
     1. [Quick Overview](#quick-overview)
     2. [Examples](#examples)
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

### Manual install
First, download the repository. Then from the `diffusion-fit` folder/directory run
```
pip install .
```

Note that `diffusion-fit` is a private repository, so need a GitHub account and
need to be part of the NTBEL organization to have access and install.

### Recommended additional software

The following software is not required for the basic operation of **diffusonfit**, but provide extra capabilities and features when installed.

#### tqdm
Command line runs will display a progress bar that tracks the analysis of the set of image files when the [tqdm](https://github.com/tqdm/tqdm) package installed.  

------

# License

This project is currently a private repository for internal NTBEL
organization use only.

------

# Change Log

See: [CHANGELOG](CHANGELOG.md)

------

# Documentation and Usage

### Quick Overview
Principally, `diffusion-fit` defines the **GaussianFit** class,
```python
from diffusonfit import GaussianFit
```
which defines an object that can be used fit the 2D diffusion distribution with
two-step fitting procedure, assuming the fluorescence from the diffusion cloud
has a Gaussian distribution whose width changes over time. The procedure is
adapted from the two-step fitting procedure described described in Nicholson and Tao 1993 [doi: 10.1016/S0006-3495(93)81324-9](https://doi.org/10.1016/S0006-3495(93)81324-9).

### Examples
```python
gfit = GaussianFit('images.tif',
                    stimulation_frame=50, timestep=0.6,
                    pixel_width=0.99, stimulation_radius=30)
Dstar = gfit.fit(verbose=True, s_to_n=3)                    
```
To see the plots corresponding to each of the two fitting steps:
```python
# Step 1
gfit.display_image_fits(saveas='step1_fits.png')
# Step 2
gfit.display_linear_fit(saveas='step2_linfit.png')
```

------

# Contact

Please open a [GitHub Issue](https://github.com/NTBEL/diffusion-fit/issues) to
report any problems/bugs or make any comments, suggestions, or feature requests.

------

# Citing

TBD
