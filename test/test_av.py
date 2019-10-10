from __future__ import annotations

import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mdtraj as md
import avtraj as avt
import numpy as np


class Tests(unittest.TestCase):

    def test_av_traj_init_calc(self):
        xtc_filename = './data/xtc/1am7_corrected.xtc'
        topology_filename = './data/xtc/1am7_protein.pdb'
        traj = md.load(
            xtc_filename,
            top=topology_filename
        )
        av_parameters = {
            'simulation_type': 'AV1',
            'linker_length': 20.0,
            'linker_width': 1.0,
            'radius1': 3.5,
            'simulation_grid_resolution': 1.0,
        }
        av_traj = avt.AVTrajectory(
            traj,
            av_parameters=av_parameters,
            name='57',
            attachment_atom_selection='resSeq 57 and name CB'
        )
        self.assertEqual(
            type(av_traj),
            avt.AVTrajectory
        )
        self.assertEqual(
            type(av_traj[0]),
            avt.AccessibleVolume
        )

    def test_save_av(self):
        # First load an MD trajectory by mdtraj
        xtc_filename = './data/xtc/1am7_corrected.xtc'
        topology_filename = './data/xtc/1am7_protein.pdb'
        traj = md.load(
            xtc_filename,
            top=topology_filename
        )
        av_traj = avt.AVTrajectory(
            traj,
            name='57',
            attachment_atom_selection='resSeq 57 and name CB'
        )
        av_traj[0].save_av()

    def test_distance(self):
        # First load an MD trajectory by mdtraj
        xtc_filename = './data/xtc/1am7_corrected.xtc'
        topology_filename = './data/xtc/1am7_protein.pdb'
        traj = md.load(
            xtc_filename,
            top=topology_filename
        )

        av_parameters_donor = {
            'simulation_type': 'AV1',
            'linker_length': 20.0,
            'linker_width': 1.0,
            'radius1': 3.5,
            'simulation_grid_resolution': 1.0,
        }

        av_traj_donor = avt.AVTrajectory(
            traj,
            av_parameters=av_parameters_donor,
            name='57',
            attachment_atom_selection='resSeq 57 and name CB'
        )

        av_parameters_acceptor = {
            'simulation_type': 'AV1',
            'linker_length': 20.0,
            'linker_width': 1.0,
            'radius1': 3.5,
            'simulation_grid_resolution': 1.0,
        }

        av_traj_acceptor = avt.AVTrajectory(
            traj,
            av_parameters=av_parameters_acceptor,
            name='136',
            attachment_atom_selection='resSeq 136 and name CB'
        )

        distances = []
        distances_fret = []
        n_frames = len(traj)
        for i in range(n_frames):
            av_d = av_traj_donor[i]
            av_a = av_traj_acceptor[i]
            distances.append(
                av_d.dRDA(av_a)
            )
            distances_fret.append(
                av_d.dRDAE(av_a)
            )

        distances_ref = np.array(
            [
                45.63678211, 46.04870158, 45.49225274, 45.66995817, 46.65077912,
                44.74206475, 42.85184406, 43.77119739, 42.2474873, 43.16160116,
                44.60830365, 44.56985815, 45.26192564, 43.64526989, 45.60707217,
                44.14293877, 45.03892491, 47.23718578, 47.15493861, 43.45212065,
                49.28761748, 46.03995011, 46.95940379, 48.04215313, 47.39585838,
                48.31693546, 49.08759384, 50.8372694, 50.04364712, 50.02173472,
                49.38929464, 49.02257673, 47.55191912, 49.08684268, 49.46213924,
                49.55160482, 49.01382552, 48.63494367, 48.9723047, 46.20087407,
                49.58790713, 51.51456704, 47.62595471, 48.9599253, 47.65023549,
                47.66830441, 48.78244785, 46.6308259, 49.16116225, 48.23696728,
                49.62011666
            ]
        )
        self.assertEqual(
            np.allclose(
                np.array(distances),
                distances_ref,
                atol=0.5
            ),
            True
        )


if __name__ == '__main__':
    unittest.main()
