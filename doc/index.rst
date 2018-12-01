Welcome to AvTraj's documentation!
==================================

AvTraj is tool to calculate accessible volumes of labels attached to biomolecules. using MD trajectories as an input
qvTraj can generate, write, and analyze accessible volumes (AVs) with only a few lines of Python code. By the use of
LabelLib AvTraj provides programmatic access to `latest developments <https://doi.org/10.1016/j.sbi.2016.11.012/>`_
in implicit dye models for FRET experiments.

AvTraj is a python library that allows users to perform simulations of accessible volumes for molecular
dynamics (MD) trajectories. AvTraj serves as a high-level interface for the development of new methodologies
for structure-based fluorescence spectroscopy.

Features include:

        A wide support of diverse MD formats by the use of MDTraj. Extremely fast calculation of AVs by the
        use of LabelLib. Extensive analysis functions including those that compute inter-dye distances,
        FRET-efficiencies, fluorescence decays, distance distributions, and an Pythonic API.

AVTraj includes a command-line application, avana, for screening and analyzing structural models.


Usage
=====

.. toctree::
   :maxdepth: 1

   installation.rst
   tutorial.rst


AV Parameters
-------------

.. toctree::
   :maxdepth: 1

   parameters.rst


API Reference
-------------

.. toctree::
   :maxdepth: 1

   api.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
