import mdtraj as md

import avtraj
from avtraj import av
import pylab as p

# First load an MD trajectory by mdtraj
traj_fn = './examples/hGBP1_out_3.h5'
traj = md.load(traj_fn)

# Pass a trajectory to fps.AVTrajectory. This creates an object, which can be
# accessed as a list. The objects within the "list" are accessible volumes
av_traj_1 = avtraj.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB', strip_mask='resSeq 7')
av_traj_1 = avtraj.AVTrajectory(traj_fn, '18D', attachment_atom_selection='resSeq 7 and name CB', strip_mask='resSeq 7')

# These accessible volumes can be saved as xyz-file
av_traj_1[0].save_xyz('test_18_0.xyz')
av0 = av_traj_1[0]

# Pass a trajectory to fps.AVTrajectory. This creates an object, which can be
# accessed as a list. The objects within the "list" are accessible volumes
av_traj_2 = avtraj.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 550 and name CB', strip_mask='resSeq 550')
# These accessible volumes can be saved as xyz-file
av1 = av_traj_2[0]
x, y = av1.pRDA(av0)
p.plot(x, y)


# The dye parameters can either be passed explicitly on creation of the object
av_traj = avtraj.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB', linker_length=25., linker_width=1.5, radius_1=6.0)

# or they can be selected from a predefined set of parameters found in the JSON file dye_definition.json located within
# the package directory
av_traj = avtraj.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB', dye_parameter_set='D3Alexa488')

# To calculate a trajectory of distances and distance distributions first a labeling file and a "distance file"
# needs to be specified. The distance file contains a set of labeling positions and distances and should be compatible
# to the labeling files used by the software "Olga"
av_dist = avtraj.AvDistanceTrajectory(traj, './examples/hGBP1_distance.json')

# For every frame the 'chi2' (for the provided set of distances), the FRET efficiency expressed in units of
# a distance 'rDAE', the distance between the mean dye positions 'rMP', and the donor-acceptor distance
# distribution 'pRDA' is calculated. 'pRDA' is a histogram. The histogram bins can be specified upon initialization
# of the AvDistanceTrajectory object and are returned as a value.
print av_dist[0]
