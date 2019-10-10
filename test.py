import mdtraj as md
import avtraj as avt
import json

# First load an MD trajectory by mdtraj
import avtraj.trajectory

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
av_traj_donor = avtraj.trajectory.AVTrajectory(
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
av_traj_acceptor = avtraj.trajectory.AVTrajectory(
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
av_dist = avtraj.trajectory.AvDistanceTrajectory(
    traj,
    json.load(
        open(
            labeling_file
        )
    )
)
print(av_dist[:10])
