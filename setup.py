import io
import os

from setuptools import setup, find_packages

# read the __version__ variable from nisqai/_version.py
exec(open("src/qmsc/_version.py").read())

# readme file as long description
long_description = ("======\n" +
                    "QMSC\n" +
                    "======\n")
stream = io.open("README.md", encoding="utf-8")
stream.readline()
long_description += stream.read()

# read in requirements.txt
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

setup(
    name="qmsc",
    version=__version__,
    author="Nic Ezzell",
    author_email="nezzell@usc.edu",
    url="https://github.com/naezzell/qmsc",
    description="Library for Quantum Mixed State Compiling (QMSC)",
    long_description=long_description,
    install_requires=requirements,
    license="None",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
    )
