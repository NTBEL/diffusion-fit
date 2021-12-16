# diffusion-fit



**diffusion-fit** is a python package for extract estimates of dye/peptide diffusion coefficients and loss rates from a time-sequence of fluorescence images.


## Table of Contents

 1. [Install](#install)
     1. [pip install](#pip-install)
     2. [conda install](#conda-install)
     3. [Recomended additional software](#recomended-additional-software)
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

Note that `diffusion-fit` has the following core dependencies:
   * [NumPy](http://www.numpy.org/)
   * [SciPy](https://www.scipy.org/)
   * [scikit-image](https://scikit-image.org/)
   * [Matplotlib](https://matplotlib.org/)
   * [seaborn](https://seaborn.pydata.org/)

### pip install
You can install the latest version of `diffusion-fit` using `pip` sourced from the GitHub repo:
```
pip install -e git+https://github.com/blakeaw/GAlibrate@v0.6.0#egg=galibrate
```
However, this will not automatically install the core dependencies. You will have to do that separately:
```
pip install numpy scipy scikit-image matplotlib seaborn
```
Note that `diffusion-fit` is a private repository, so need a GitHub account and
need to be part of the NTBEL organization to have access and install.

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
Principally, **GAlibrate** defines the **GAO** (continuous **G**enetic **A**lgorithm-based **O**ptimizer ) class,
```python
from galibrate import GAO
```
which defines an object that can be used setup and run a continuous genetic algorithm-based optimization (i.e., a maximization) of a user-defined fitness function over the search space of a given set of (model) parameters.


### Examples


------

# Contact

Please open a [GitHub Issue](https://github.com/NTBEL/diffusion-fit/issues) to
report any problems/bugs or make any comments, suggestions, or feature requests.

------

# Citing

TBD
