from __future__ import annotations

import utils
import os
import unittest


TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import json
import mdtraj as md
import avtraj
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
            'residue_seq_number': 57,
            'atom_name': 'CA'
        }
        av_traj = avtraj.trajectory.AVTrajectory(
            traj,
            av_parameters=av_parameters,
            name='57'
        )
        self.assertEqual(
            type(av_traj),
            avtraj.trajectory.AVTrajectory
        )
        self.assertEqual(
            type(av_traj[0]),
            avtraj.av.AccessibleVolume
        )

    def test_save_av(self):
        # First load an MD trajectory by mdtraj
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
            'residue_seq_number': 57,
            'atom_name': 'CA'
        }
        av_traj = avtraj.trajectory.AVTrajectory(
            traj,
            name='57',
            av_parameters=av_parameters
        )
        av_traj[0].save_av()

    def test_distance(self):
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
            'residue_seq_number': 57,
            'atom_name': 'CA'
        }

        av_traj_donor = avtraj.trajectory.AVTrajectory(
            traj,
            av_parameters=av_parameters_donor,
            name='57',
        )

        av_parameters_acceptor = {
            'simulation_type': 'AV1',
            'linker_length': 20.0,
            'linker_width': 1.0,
            'radius1': 3.5,
            'simulation_grid_resolution': 1.0,
            'residue_seq_number': 136,
            'atom_name': 'CA'
        }

        av_traj_acceptor = avtraj.trajectory.AVTrajectory(
            traj,
            av_parameters=av_parameters_acceptor,
            name='136'
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
            [46.58445179, 46.10678172, 46.64247226, 46.97069943, 47.7507035,
             45.31224551,
             44.55804641, 45.59746112, 44.4557364,  45.01399551, 45.40064313,
             45.90336265,
             46.12118593, 44.55974777, 46.4193657,  44.15305281, 45.92528762,
             47.52961313,
             47.49590247, 43.94771434, 49.33894502, 47.72217377, 47.80850216,
             48.37280611,
             47.59650977, 48.98418512, 49.41689837, 51.25695952, 50.29100387,
             50.37597367,
             49.90922226, 49.45598574, 48.19644809, 49.9073324,  50.00831528,
             49.72726928,
             49.52950194, 48.86970621, 49.33807575, 47.00636766, 49.9563316,
             51.46870868,
             48.05152249, 49.60495012, 48.17830894, 48.3569286,  49.35266764,
             47.42559334,
             49.46402385, 48.52772225, 49.79250257]
        )
        self.assertEqual(
            np.allclose(
                np.array(distances),
                distances_ref,
                atol=1.0
            ),
            True
        )

    def test_labeling_file(self):
        xtc_filename = './data/xtc/1am7_corrected.xtc'
        topology_filename = './data/xtc/1am7_protein.pdb'
        traj = md.load(
            xtc_filename,
            top=topology_filename
        )

        labeling_file = './data/labeling.fps.json'
        av_dist = avtraj.trajectory.AvDistanceTrajectory(
            traj,
            json.load(
                open(
                    labeling_file
                )
            )
        )
        d_ref = {
            '158_57': {
                'rMP': [
                    43.829381736544605, 44.10156625284363,
                    43.967130339424806, 43.82132925834934,
                    45.414706914860034, 41.76623812423402,
                    41.43372888588546, 42.44166006076308,
                    41.23644763594793, 41.826784810818715
                ], 'rDA': [
                    47.11416931661129, 47.388952173485755,
                    47.218177980394366, 47.05746754247665,
                    48.591604348430636, 45.213844204669,
                    45.006189569416044, 45.826895953736305,
                    44.69403249952793, 45.31119674283981
                ], 'rDAE': [
                    48.05623254175495, 48.297890696899145,
                    48.093081710691074, 47.945264281224695,
                    49.14637697087784, 46.648591198310264,
                    46.52195315168675, 47.166464042318125,
                    46.23803209414695, 46.77764388100564
                ], 'chi2': [
                    5.3384868656373206, 5.46931002909805,
                    4.088271923570988, 4.479691168548389,
                    4.331099480742713, 4.652876099989579,
                    5.742968026665894, 3.7369019221906656,
                    3.367216724426841, 4.533413274090362
                ]},
            '57_136': {
                'rMP': [
                    43.829381736544605, 44.10156625284363,
                    43.967130339424806, 43.82132925834934,
                    45.414706914860034, 41.76623812423402,
                    41.43372888588546, 42.44166006076308,
                    41.23644763594793, 41.826784810818715
                ],
                'rDA': [
                    47.11416931661129, 47.388952173485755,
                    47.218177980394366, 47.05746754247665,
                    48.591604348430636, 45.213844204669,
                    45.006189569416044, 45.826895953736305,
                    44.69403249952793, 45.31119674283981
                ], 'rDAE': [
                    48.05623254175495, 48.297890696899145,
                    48.093081710691074, 47.945264281224695,
                    49.14637697087784, 46.648591198310264,
                    46.52195315168675, 47.166464042318125,
                    46.23803209414695, 46.77764388100564
                ], 'chi2': [
                    5.3384868656373206, 5.46931002909805,
                    4.088271923570988, 4.479691168548389,
                    4.331099480742713, 4.652876099989579,
                    5.742968026665894, 3.7369019221906656,
                    3.367216724426841, 4.533413274090362
                ]
            }
        }

        d_m = av_dist[:10]
        for key_label in d_ref:
            for key_v in d_ref[key_label]:
                self.assertEqual(
                    np.allclose(
                        d_m[key_label][key_v],
                        d_ref[key_label][key_v],
                        0.4
                    ),
                    True
                )


if __name__ == '__main__':
    unittest.main()
