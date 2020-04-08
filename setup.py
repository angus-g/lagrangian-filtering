from setuptools import setup, find_packages

setup(
    name="lagrangian-filtering",
    description="Temporal filtering of data in a Lagrangian frame of reference",
    author="Angus Gibson",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    extras_require={"build": ["pytest", "xarray", "cloudpickle"]},
    install_requires=[
        "dask[array]",
        "h5py",
        "numpy>=1.17.0",
        "scipy>=1.2.0",
        "xarray",
        "netCDF4",
        "cftime",
        "parcels @ git+https://github.com/angus-g/parcels@lagrangian-filtering",
    ],
)
