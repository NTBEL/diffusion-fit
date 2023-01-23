import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diffusionfit",
    version="0.8.0",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "seaborn",
        "pandas",
        "numba",
        "streamlit",
        "plotly",
    ],
    author="Blake A. Wilson",
    author_email="blake.wilson@utdallas.edu",
    description="Python package for extract estimates of dye/peptide diffusion coefficients and loss rates from a time-sequence of fluorescence images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NTBEL/diffusion-fit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
