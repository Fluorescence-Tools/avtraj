Welcome to AvTraj's documentation!
==================================

AvTraj is tool to calculate FRET observables from MD-trajectories. Read, write and analyze accessible volumes (AVs)
using MD trajectories as an input with only a few lines of Python code. By the use of LabelLib AvTraj provides
programmatic access to latest developments in implicit dye models for FRET experiments [![DOI for Citing COSB](https://img.shields.io/badge/DOI-10.1016/j.sbi.2016.11.012-blue.svg)](https://doi.org/10.1016/j.sbi.2016.11.012).

AvTraj is a python library that allows users to perform simulations of accessible volumes for molecular
dynamics (MD) trajectories. AvTraj serves as a high-level interface for the development of new methodologies
for structure-based fluorescence spectroscopy.

Features include:

        A wide support of diverse MD formats by the use of MDTraj. Extremely fast calculation of AVs by the
        use of LabelLib (e.g. xxxx the speed of yyyy). Extensive analysis functions including those that compute
        inter-dye distances, FRET-efficiencies, fluorescence decays, distance distributions, and an Pythonic API.

AVTraj includes a command-line application, avana, for screening and analyzing structural models.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

.. automodule:: avtraj
   :members:

.. automodule:: avtraj.av_functions
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
