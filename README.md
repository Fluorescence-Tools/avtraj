AvTraj
======

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


Relation of other software and libraries
----------------------------------------

LabelLib serves as core low-level library for the software Olga and the higher-level Python library AvTraj. The
deprecated software FPS is independent of LabelLib.

![LabelLib and other software/libraries][3]

[Olga](https://github.com/Fluorescence-Tools/Olga) is a software dedicated towards experimentalists. Olga provides a graphical user interface for the calculation of accessible volumes (AVs), screen a set of structural models against experimental observables, rigid-body docking, 
and the optimal design of new FRET experiments. 

[AvTraj](https://github.com/Fluorescence-Tools/avtraj)
AvTraj is a Python library for the calculation of accessible volumes (AVs), screening. AvTraj facilitates the development of new analytical approaches for FRET-based structural models. Avtraj facilitates processing of 
MD-simulations and the development of Python scripts handling FRET-based structural models. 

[FPS](http://www.mpc.hhu.de/software/fps.html) is a software with a graphical user interface for the FRET-based structural modeling. FPS can calculate accessible volumes (AVs), screen a set of structural models against experimental observables, and can generate new structural 
models by rigid-body docking using experimental FRET data.


Installation
============

Anaconda
--------

```commandline
conda --add channels tpeulen
conda install avtraj
```


Code Example
============

```python
import mdtraj as md
import avtraj as avt

# First load an MD trajectory by mdtraj
traj = md.load('./examples/hGBP1_out_3.h5')

# Pass a trajectory to fps.AVTrajectory. This creates an object, which can be 
# accessed as a list. The objects within the "list" are accessible volumes  
av_traj = avt.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB')
# These accessible volumes can be saved as xyz-file
av_traj[0].save_xyz('test_344.xyz')

# The dye parameters can either be passed explicitly on creation of the object
av_traj = avt.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB', linker_length=25., linker_width=1.5, radius_1=6.0)

# or they can be selected from a predefined set of parameters found in the JSON file dye_definition.json located within
# the package directory 
av_traj = avt.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB', dye_parameter_set='D3Alexa488')

# To calculate a trajectory of distances and distance distributions first a labeling file and a "distance file" 
# needs to be specified. The distance file contains a set of labeling positions and distances and should be compatible
# to the labeling files used by the software "Olga". By default the 
av_dist = avt.AvDistanceTrajectory(traj, './examples/hGBP1_distance.json')

```


Citations 
=========

* MDTraj - [![DOI for Citing MDTraj](https://img.shields.io/badge/DOI-10.1016%2Fj.bpj.2015.08.015-blue.svg)](http://doi.org/10.1016/j.bpj.2015.08.015)
* FPS - [![DOI for Citing FPS](https://img.shields.io/badge/DOI-10.1038/nmeth.2222-blue.svg)](http://doi.org/10.1038/nmeth.2222)


License
=======

GNU LGPL version 2.1, or at your option a later version of the license.
Various sub-portions of this library may be independently distributed under
different licenses. See those files for their specific terms.

[3]: doc/img/software_overview.svg "LabelLib and other software/libraries"
