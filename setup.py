from setuptools import setup, find_packages

setup(
    name="pyae",
    version="0.0.0a5",  # formato válido para pre-releases
    author="Tu Nombre",
    description="PyTorch Autoencoder framework for EMI signal analysis",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5",
        "matplotlib>=3.5",
        "plotly>=5.0",
        "statsmodels>=0.13",
        "torch>=1.12",
        "openpyxl>=3.0",
        "numpy>=1.21",
        "ipython",
        "kaleido",
        "torchinfo>=1.8",
        "torchmetrics>=0.11",
        # "h5py",      # descomenta si lo usas
        # "xarray[complete]",
        # "netCDF4",   # descomenta si lo usas
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
