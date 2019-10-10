from __future__ import annotations
from typing import Dict

import mdtraj
import numpy as np

from .base import PythonBase
from . import av_functions
from .av import AccessibleVolume


class AVTrajectory(PythonBase):
    """Calculates for a an MD trajectory a corresponding trajectory of
    accessible volumes AccessibleVolume.


    Attributes
    ----------

    trajectory : mdtraj trajectory

    position_name : position name


    Examples
    --------
    Make a new accessible volume trajectory. This places a dye on the
    specified atom. The dye parameters are either passes as or taken from
    a dye-library (see dye_definition.json). If no dye-parameters are passed
    default parameters are used (not recommended).
    For visual inspection the accessible volume can be saved as xyz-file.

    >>> import mdtraj as md
    >>> import avtraj as avt
    >>> traj = md.load('./doc/examples/traj.h5')

    >>> av_parameters = dict()
    >>> av_parameters['chain_identifier'] = 'A'
    >>> av_parameters['residue_seq_number'] = 344
    >>> av_parameters['atom_name'] = 'CB'
    >>> av_parameters['linker_length'] = 20.0
    >>> av_parameters['linker_width'] = 1.5
    >>> av_parameters['radius1'] = 3.5
    >>> av_parameters['strip_mask'] = "MDTraj: residue 344 and not (name CA or name C or name N or name O)"
    >>> av_parameters['contact_volume_thickness'] = 6.5
    >>> av_parameters['contact_volume_trapped_fraction'] = 0.8
    >>> av_parameters['simulation_type'] = 'AV1'
    >>> av_parameters['simulation_grid_resolution'] = 0.5
    >>> av_parameters['label_interaction_sites'] = [{"selection": "MDTraj: resSeq 344", "weight": 1.0, "radius": 9.0}]
    >>> av_traj = avt.AVTrajectory(traj, 'test', av_parameters=av_parameters)
    >>> av = av_traj[0]
    >>> av.save_av(filename='test_344', openDX=True)

    All arguments are compatible to the JSON file format also supported by the software Olga.
    >>> import json
    >>> lf = json.load(open('./doc/examples/labeling.json', 'r'))
    >>> av_traj = avt.AVTrajectory(traj, 'test', av_parameters=lf['Positions']['344A'])
    >>> av_traj[0].save_av(filename='test_344', openDX=True)

    """

    def __init__(
            self,
            traj: mdtraj.Trajectory,
            name: str,
            av_parameters: Dict,
            attachment_atom_selection: str = None,
            **kwargs
    ):
        """
        Parameters
        ----------
        name : str

        traj : mdtraj Trajectory object or str
            Either a mdtraj Trajectory object or a string containing a the path
            to a trajectory file.
        top : str (optional)
            The filename of a topology file. This options is only used if the
            traj parameter is an string.
        av_parameters : dict (optional)
            A dictionary containing the parameters used to define an AV. The
            names of the parameters are described in the JSON file format file.
            If no parameters are provided the AV is initialized using the
            following default parameters::
                av_parameters = {
                    'simulation_type': 'AV1',
                    'linker_length': 20.0,
                    'linker_width': 1.0,
                    'radius1': 3.5,
                    'simulation_grid_resolution': 0.5,
                }
        cache_avs : bool (optional)
            If cache_avs is True the AVs are stored in an dictionary where the
            keys of the dictionary correspond to the frame numbers to prevent
            excessive recalculations. If this parameter is False the AVs for a
            frame are recalculated and not stored.
        attachment_atom_selection: str (optional)
            This is a MDTraj selection string that is used to define the
            attachment atom. If this selection string is not provided,
            the attachment atom is determined using the parameters provided
            in the av_parameters dictionary.


        """
        kwargs['name'] = name
        kwargs['verbose'] = kwargs.pop('verbose', False)
        av_parameters['labeling_site_name'] = name
        kwargs['av_parameters'] = av_parameters
        kwargs['cache_avs'] = kwargs.get('cache_avs', True)
        super().__init__(self, **kwargs)

        # Load the trajectory or use the passed mdtraj Trajectory object
        if isinstance(traj, str):
            top = kwargs.pop('top', None)
            self.trajectory = mdtraj.load(traj, top=top) if isinstance(traj, str) else traj
        else:
            if not isinstance(traj, mdtraj.Trajectory):
                raise TypeError(
                    'The passed trajectory parameter traj needs to be either a string '
                    'pointing to a trajectory file or an mdtraj Trajectory object.'
                )
            self.trajectory = traj

        self._avs = dict()
        self.vdw = kwargs.get('vdw', av_functions.get_vdw(self.trajectory))

        selection_residue = "residue " + str(
            av_parameters['residue_seq_number']
        )
        if isinstance(attachment_atom_selection, str):
            selection = attachment_atom_selection
        else:
            # Determine attachment atom index
            selection = ""
            selection += selection_residue
            chain_id = av_parameters.get('chain_identifier', '').lower()
            try:
                selection += " and chainid " + av_functions.LETTERS[
                        chain_id
                ]
            except KeyError:
                print("Not a valid chain ID")
            selection += " and name " + str(av_parameters['atom_name'])
        attachment_atom_index = self.trajectory.topology.select(selection)[0]
        self.attachment_atom_index = attachment_atom_index

        # Apply "strip_mask" and change vdw-radii of atoms selected by
        # the strip mask to zero.
        strip_mask = av_parameters.get(
            'strip_mask', "MDTraj: " + selection_residue
        )
        t, sm = strip_mask.split(': ')
        if t == "MDTraj":
            strip_mask_atoms = self.trajectory.topology.select(sm)
        else:
            print(
                'WARNING: Only MDTraj selections are allowed as strip mask'
            )
            strip_mask_atoms = list()
        self.vdw[strip_mask_atoms] *= 0.0

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, key):
        if isinstance(key, int):
            frame_idx = [key]
        elif isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            frame_idx = range(start, min(stop, len(self)), step)
        else:
            raise ValueError("")

        re = []
        for frame_i in frame_idx:
            frame = self.trajectory[frame_i]

            # If an AV was already calculated use pre-calculated av
            if frame_i in self._avs.keys() and self.cache_avs:
                av = self._avs[frame_i]
            else:
                parameters = self.av_parameters
                parameters['labeling_site_name'] += "_" + str(frame_i)
                xyz = (frame.xyz[0] * 10.0).astype(np.float64)
                vdw = self.vdw
                xyzr = np.vstack([xyz.T, vdw])
                topology = self.trajectory.topology
                r1 = parameters.get('radius1', 0.0)
                r2 = parameters.get('radius2', 0.0)
                r3 = parameters.get('radius3', 0.0)
                inter_action_radius = max(r1, r2, r3)

                # compile array xyzrq of acv atoms with radii and weights
                try:
                    label_interaction_sites = parameters['label_interaction_sites']
                    xyzrq = list()
                    for interaction_site in label_interaction_sites:
                        selection_type, selection = interaction_site["selection"].split(": ")
                        weight = interaction_site["weight"]
                        radius = interaction_site["radius"]
                        if selection_type == "MDTraj":
                            ai = topology.select(selection)
                        else:
                            raise AttributeError(
                                'Only MDTraj selections are allowed as strip mask'
                            )
                        if len(ai) > 0:
                            xyzi = xyz[ai]
                            ri = vdw[ai] + radius + inter_action_radius
                            wi = np.zeros_like(ri) + float(weight)
                            xyzrq.append(np.vstack([xyzi.T, ri, wi]).T)
                            xyzrq_array = np.vstack(xyzrq).T
                        else:
                            xyzrq_array = []
                except KeyError:
                    xyzrq_array = []
                parameters['interaction_sites_xyzrq'] = xyzrq_array

                attachment_coordinate = xyz[self.attachment_atom_index]
                av = AccessibleVolume(
                    xyzr,
                    attachment_coordinate,
                    **parameters
                )
                self._avs[frame_i] = av

            re.append(av)

        if len(re) == 1:
            return re[0]
        else:
            return re


class AvDistanceTrajectory(
    PythonBase
):
    """
    The AvPotential class provides the possibility to calculate the reduced
    or unreduced chi2 given a set of labeling positions and experimental
    distances. Here the labeling positions and distances are provided as
    dictionaries.

    Examples
    --------

    distance_file = './examples/hGBP1_distance.json'
    av_dist = mdtraj.fluorescence.fps.AvDistanceTrajectory(traj, distance_file)
    av_dist[:3]

    >>> import avtraj as avt
    >>> import mdtraj as md
    >>> import json
    >>> traj = md.load('./doc/examples/traj.h5')
    >>> labeling = json.load(open('./doc/examples/labeling.json', 'r'))
    >>> atj = avt.AvDistanceTrajectory(trajectory=traj, labeling=labeling)

    """

    def __init__(self, trajectory, labeling, **kwargs):
        kwargs['labeling'] = labeling
        super().__init__(**kwargs)
        self.distances = labeling['Distances']
        self.positions = labeling['Positions']
        self.trajectory = trajectory

        # Initialize the AV trajectories with the parameters provided by the
        # labeling dictionary
        arguments = [
            dict(
                {
                    'traj': trajectory,
                    'name': position_key,
                    'av_parameters': self.positions[position_key]
                },
            )
            for position_key in self.positions
        ]
        self._d = dict()
        self.avs = dict(
            zip(
                self.positions.keys(),
                map(lambda x: AVTrajectory(**x), arguments)
            )
        )

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, key):

        if isinstance(key, int):
            frame_idx = [key]
        else:
            start = 0 if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            frame_idx = range(start, min(stop, len(self)), step)

        re = dict(
            (
                key,
                {'rMP': [], 'rDA': [], 'rDAE': [], 'chi2': []}
            ) for key in self.distances.keys()
        )
        for frame_i in frame_idx:
            # Don't repeat calculations
            if frame_i in self._d.keys():
                rDA, rDAE, rMP, chi2 = self._d[frame_i]
            else:
                # calculate the AVs of the frame
                avs = self.avs
                for distance_key in self.distances:
                    distance = self.distances[distance_key]
                    av1 = avs[distance['position1_name']][frame_i]
                    av2 = avs[distance['position2_name']][frame_i]
                    R0 = distance['Forster_radius']

                    rMP = av_functions.dRmp(av1, av2)
                    rDAE = av1.dRDAE(av2, forster_radius=R0)
                    rDA = av1.dRDA(av2)

                    if self.verbose:
                        print("RDA: %s" % rDA)
                        print("RDA_E: %s" % rDAE)
                        print("RDA_mp: %s" % rMP)

                    if self.distances[distance_key]['distance_type'] == 'RDAMean':
                        self.distances[distance_key]['model_distance'] = rDA
                    elif self.distances[distance_key]['distance_type'] == 'RDAMeanE':
                        self.distances[distance_key]['model_distance'] = rDAE
                    elif self.distances[distance_key]['distance_type'] == 'Rmp':
                        self.distances[distance_key]['model_distance'] = rMP

                # compare to experiment: calculate chi2
                chi2 = 0.0
                for distance in list(self.distances.values()):
                    dm = distance['model_distance']
                    de = distance['distance']
                    error_neg = distance['error_neg']
                    error_pos = distance['error_pos']
                    d = dm - de
                    chi2 += (d / error_neg) ** 2 if d < 0 else (d / error_pos) ** 2
                self._d[frame_i] = (rDA, rDAE, rMP, chi2)

            for distance_key in self.distances:
                re[distance_key]['rMP'].append(rMP)
                re[distance_key]['rDA'].append(rDA)
                re[distance_key]['rDAE'].append(rDAE)
                re[distance_key]['chi2'].append(chi2)

        return re