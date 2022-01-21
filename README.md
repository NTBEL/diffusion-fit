# diffusion-fit



**diffusion-fit** is a python package that defines models to fit the 2D fluorescence intensity distribution in a time lapse series of fluorescence microscope images and extract estimates of a diffusion coeffient using a point-source paradigm.  

The underlying point-source paradigm and two-step fitting procedure employed by the models is adapted from the methods used to analyze integrative optical imaging data ([Nicholson and Tao 1993](https://doi.org/10.1016/S0006-3495(93)81324-9) and [Hrabe and Hrabetova 2019](https://doi.org/10.1016/j.bpj.2019.08.031)). **diffusion-fit** therefore provides a model for fitting 2D imaging data for fluorescent dyes released from a point-like source as in integrative optical imaging ([Nicholson and Tao 1993](https://doi.org/10.1016/S0006-3495(93)81324-9)) or nanovesicle photorelease ([Xiong et al. 2021](https://doi.org/10.1101/2021.09.10.459853)) experiments.

Additionally, we have extended the framework to provide a model for fitting data from receptor-based fluorescent peptide sensors, such as dLight ([Patriarchi et al. 2018](https://dx.doi.org/10.1126/science.aat4422)) and kLight ([Abraham et al. 2021](https://doi.org/10.1038/s41386-021-01168-2)), when the target peptides are released from a point-like source as in nanovesicle photorelease ([Xiong et al. 2021](https://doi.org/10.1101/2021.09.10.459853)) experiments.
**diffusion-fit** also further extends the point-source paradigm to allow for additional fitting and estimation of first order loss rate constants from the 2D imaging data, which are particularly relevant for peptide volume transmission in the brain ([Xiong et al. 2021](https://doi.org/10.1101/2021.09.10.459853)).  

### What's new in

#### version 0.3.0
 * PointClarkFit fitting class for fitting the fluorescent signal of receptor-based peptide sensors during peptide point source diffusion.
 * Funtionality to estimate the loss rate of the diffusing species.
 * Additional options when running **diffusion-fit** with it's command line interface.
 * Functionality to export fitting data to csv and tiff files.

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
   * [Numba](https://numba.pydata.org/)

### pip install
You can install `diffusionfit` with `pip` sourced from the GitHub repo:
```
pip install -e git+https://github.com/NTBEL/diffusion-fit#egg=diffusionfit
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
Currently, `diffusionfit` defines the **GaussianFit** and **PointClarkFit** classes,
```python
from diffusonfit import GaussianFit, PointClarkFit
```
which define models to fit the 2D fluorescence intensity distribution and estimate the diffusion coeffient of
molecules released from a point-like source.

  * **GaussianFit** assumes the fluorescence signal comes directly from the diffusing species and that the the diffusion cloud has an isotropic Gaussian distribution that expands in width over time as in the point source diffusion model.
  * **PointClarkFit** assumes the fluorescent signal comes from the diffusing species binding to a fluorescent receptor-based sensor such as the dLight or kLight peptide sensors. The diffusion is still assumed to follow the isotropic point source model but the resulting fluorescent signal follows the Clark equation for receptor-response with the diffusing molecule as the ligand and the fluorescent sensor as the receptor.

In both cases, the 2D fluorescence distribution of each image in the time lapse of fluorescence images is fitted with the 2D intensity model and then the diffusion coefficient is extracted in a second linear fitting step. This two-step fitting procedure was adpated from the methods used to analyze integrative optical imaging data as described in Nicholson and Tao 1993 [doi: 10.1016/S0006-3495(93)81324-9](https://doi.org/10.1016/S0006-3495(93)81324-9) and Hrabe and Hrabetova 2019 [doi: 10.1016/j.bpj.2019.08.031](https://doi.org/10.1016/j.bpj.2019.08.031).

`diffusionfit` can be used both programatically or from the command line. The command line interface simplifies the process for quickly initiating analysis of multiple image files (tiff files), and it automatically generates plot images of the fits and exports the fitting parameters to a csv file for further analysis for each image file (dumped into a new output fold named diffusion_fitting). However, programmatic use offers more flexibility for generating customized fitting and analysis workflows.   

### Examples

#### Programmatic use
```python
# Units:
# timestep (s)
# pixel_width (micron)
# stimulation_radius (micron)
gfit = GaussianFit('images.tif',
                    stimulation_frame=50, timestep=0.6,
                    pixel_width=0.99, stimulation_radius=30)
Dstar = gfit.fit(verbose=True, step1_threshold=3)                    
```
To see the plots corresponding to each of the two fitting steps:
```python
# Step 1
gfit.display_image_fits(saveas='step1_fits.png')
# Step 2
gfit.display_linear_fit(saveas='step2_linfit.png')
```
#### Command line use
```
python -m diffusionfit 0.25 2.486 50 75 --loss-rate
```
To show the usage and the full set of command line options:
```
python -m diffusionfit --help
```

------

# Contact

Please open a [GitHub Issue](https://github.com/NTBEL/diffusion-fit/issues) to
report any problems/bugs or make any comments, suggestions, or feature requests.

------

# Citing
If this package is useful in your research, please cite this GitHub repo: https://github.com/NTBEL/diffusion-fit
