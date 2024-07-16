from setuptools import setup, find_packages

# Versioning SEMVER
#https://semver.org/

setup(
    name="pyae",
    version="0.0.0.a5", # version code: MAJOR.MINOR.PATCH.preReleaseStateN
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "plotly",
        "statsmodels",
        "torch",
        "openpyxl",
        "numpy",
        "h5py",
        "xarray[complete]",
        "ipython",
        "kaleido",
        "netcdf4",
        "ffmpeg",
        "torchinfo",
	    "torchsummary",
        "torchmetrics",
    ],  # List dependencies here if any
)
