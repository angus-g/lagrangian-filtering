from setuptools import setup, find_packages

setup(
    name="lagrangian-filtering",
    description="Temporal filtering of data in a Lagrangian frame of reference",
    author="Angus Gibson",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    extras_require={"build": ["pytest"]},
)
