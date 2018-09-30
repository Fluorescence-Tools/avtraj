import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="avtraj",
    version="0.0.6",
    author="Thomas-Otavio Peulen",
    url="https://github.com/Fluorescence-Tools/avtraj",
    author_email="thomas.otavio.peulen@gmail.com",
    description=("A library to calculate FRET observables for MD trajectories by accessible volume (AV) simulations."
                 "In the AV simulations the sterically allowed conformation space of the labels is approximated "
                 "the conformational space of flexible attached ellipsoids."
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
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
    package_data={
        'avtraj': ['avtraj/*.json'],
        'avtraj.examples': ['avtraj/examples/*.*']
    },
)



