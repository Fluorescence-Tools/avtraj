import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="avtraj",
    version="0.0.3",
    author="Thomas-Otavio Peulen",
    url="https://github.com/Fluorescence-Tools/avtraj",
    author_email="thomas.otavio.peulen@gmail.com",
    description=("A tool to calculate FRET observables for MD trajectories by accessible volume calculations."
                 "Here, spatial density of flexible coupled dyes/labels is approximated by the sterically allowed"
                 "space for labels modeled by ellipsoids attached by a flexible cylinder."
                 ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="LGPLv2.1",
    install_requires=[
        'LabelLib',
        'mdtraj',
        'numba'
    ],
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2",
        "Operating System :: OS Independent",
    ),
)



