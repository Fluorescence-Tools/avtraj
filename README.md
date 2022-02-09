# AvTraj
[![Build Status](https://travis-ci.org/Fluorescence-Tools/avtraj.svg?branch=master)](https://travis-ci.org/Fluorescence-Tools/avtraj)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/146004a9bd4a4c19b2fd55b8c3d10182)](https://www.codacy.com/manual/tpeulen/avtraj?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Fluorescence-Tools/avtraj&amp;utm_campaign=Badge_Grade)
[![Anaconda-Server Badge](https://anaconda.org/tpeulen/avtraj/badges/installer/conda.svg)](https://conda.anaconda.org/tpeulen)
[![Anaconda-Server Badge](https://anaconda.org/tpeulen/avtraj/badges/platforms.svg)](https://anaconda.org/tpeulen/avtraj)

AvTraj is tool to calculate FRET observables from MD-trajectories. Read
, write and analyze accessible volumes (AVs) using MD trajectories as an
input with only a few lines of Python code. By the use of LabelLib AvTraj
provides programmatic access to latestdevelopments in implicit dye models
for FRET experiments [![DOI for Citing COSB](https://img.shields.io/badge/DOI-10.1016/j.sbi.2016.11.012-blue.svg)](https://doi.org/10.1016/j.sbi.2016.11.012). 

AvTraj is a python library that allows users to perform simulations of
accessible volumes for molecular dynamics (MD) trajectories. AvTraj serves
as a high-level interface for the development of new methodologies for
 structure-based fluorescence spectroscopy.

Features include:

        A wide support of diverse MD formats by the use of MDTraj. Extremely
         fast calculation of AVs by the use of LabelLib (e.g. xxxx the speed
          of yyyy). Extensive analysis functions including those that compute
        inter-dye distances, FRET-efficiencies, fluorescence decays, distance
         distributions, and an Pythonic API.

AVTraj includes a command-line application, avana, for screening and
 analyzing structural models.

## Relation of other software and libraries
LabelLib serves as core low-level library for the software Olga and the
higher-level Python library AvTraj. The deprecated software FPS is
independent of LabelLib.

![LabelLib and other software/libraries][3]

[Olga](https://github.com/Fluorescence-Tools/Olga) is a software dedicated
 towards experimentalists. Olga provides a graphical user interface for the
  calculation of accessible volumes (AVs), screen a set of structural models
   against experimental observables, rigid-body docking, 
and the optimal design of new FRET experiments.

[AvTraj](https://github.com/Fluorescence-Tools/avtraj)
AvTraj is a Python library for the calculation of accessible volumes (AVs
), screening. AvTraj facilitates the development of new analytical approaches
for FRET-based structural models. Avtraj facilitates processing of 
MD-simulations and the development of Python scripts handling FRET-based
 structural models. 

[FPS](http://www.mpc.hhu.de/software/fps.html) is a software with a graphical
user interface for the FRET-based structural modeling. FPS can calculate
accessible volumes (AVs), screen a set of structural models against
experimental observables, and can generate new structural 
models by rigid-body docking using experimental FRET data.

## Installation
### Anaconda

The software depends on other libraries provided by conda-forge. Thus, as
a first step, the [conda-forge](https://conda-forge.org/)
channel needs to be added.

```commandline
conda --add channels conda-forge
```

In a second step, the avtraj package can be installed.

```commandline
conda --add channels tpeulen
conda install avtraj
```

## Code Example
```python
    import mdtraj as md
    import avtraj as avt
    import json
    
    # First load an MD trajectory by mdtraj
    xtc_filename = './test/data/xtc/1am7_corrected.xtc'
    topology_filename = './test/data/xtc/1am7_protein.pdb'
    traj = md.load(
        xtc_filename,
        top=topology_filename
    )
    # Define your accessible volume (AV) parameters
    av_parameters_donor = {
        'simulation_type': 'AV1',
        'linker_length': 20.0,
        'linker_width': 1.0,
        'radius1': 3.5,
        'simulation_grid_resolution': 1.0,
        'residue_seq_number': 57,
        'atom_name': 'CA'
    }
    
    # Pass a trajectory to fps.AVTrajectory. This creates an object, which can be
    # accessed as a list. The objects within the "list" are accessible volumes
    av_traj_donor = avt.AVTrajectory(
        traj,
        av_parameters=av_parameters_donor,
        name='57'
    )
    # These accessible volumes can be saved as xyz-file
    av_traj_donor[0].save_av()
    
    av_parameters_acceptor = {
        'simulation_type': 'AV1',
        'linker_length': 20.0,
        'linker_width': 1.0,
        'radius1': 3.5,
        'simulation_grid_resolution': 1.0,
        'residue_seq_number': 136,
        'atom_name': 'CA'
    }
    
    # The dye parameters can either be passed explicitly on creation of the object
    av_traj_acceptor = avt.AVTrajectory(
        traj,
        av_parameters=av_parameters_acceptor,
        name='136'
    )
    av_traj_acceptor[0].save_av()
    
    
    # To calculate a trajectory of distances and distance distributions first a
    # labeling file and a "distance file" needs to be specified. The distance
    # file contains a set of labeling positions and distances and should be
    # compatible to the labeling files used by the software "Olga".
    # or by the tool `label_structure` provided by the software ChiSurf.
    labeling_file = './test/data/labeling.fps.json'
    av_dist = avt.AvDistanceTrajectory(
        traj,
        json.load(
            open(
                labeling_file
            )
        )
    )
    print(av_dist[:10])
```

## Citations 
* MDTraj - [![DOI for Citing MDTraj](https://img.shields.io/badge/DOI-10.1016%2Fj.bpj.2015.08.015-blue.svg)](http://doi.org/10.1016/j.bpj.2015.08.015)
* FPS - [![DOI for Citing FPS](https://img.shields.io/badge/DOI-10.1038/nmeth.2222-blue.svg)](http://doi.org/10.1038/nmeth.2222)

## License
GNU LGPL version 2.1, or at your option a later version of the license.
Various sub-portions of this library may be independently distributed under
different licenses. See those files for their specific terms.

[3]: doc/img/software_overview.svg "LabelLib and other software/libraries"
