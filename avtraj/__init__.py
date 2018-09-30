# -*- coding: utf-8 -*-
"""
"""

from collections import OrderedDict
import os
import json

import mdtraj
import numpy as np
import yaml

import LabelLib as ll
from avtraj import av_functions


package_directory = os.path.dirname(__file__)
dye_definition = json.load(open(os.path.join(package_directory, 'dye_definition.json')))
dye_names = dye_definition.keys()


class PythonBase(object):
    """The class PythonBase provides base functionality to load and save keyword
    arguments used to initiate the class as a dict. Moreover, the __setstate__
    and the __getstate__ magic methods are implemented for basic functionality
    to recover and copy states of objects. Parameters passed as keyword arguments
    can be directly accesses as an attribute of the object.

    Examples
    --------
    >>> pb = PythonBase(name='name_a', lo="uu")
    >>> pb
    {'lo': 'uu', 'name': 'name_a'}
    >>> bc = PythonBase(parameter="ala", lol=1)
    >>> bc.lol
    1
    >>> bc.parameter
    ala
    >>> bc.to_dict()
    {'lol': 1, 'parameter': 'ala', 'verbose': False}
    >>> bc.from_dict({'jj': 22, 'zu': "auf"})
    >>> bc.jj
    22
    >>> bc.zu
    auf

    Methods
    -------

    save(self, filename, file_type='json')
        Saves the class either to a JSON file (file_type='json') or a YAML (file_type='yaml')
        depending on the the optional parameter 'file_type'.

    load(self, filename, file_type='json')
        Loads a JSON or YAML file to the class

    to_dict(self)
        Returns a python dictionary representing the state of the object

    to_json(self, indent=4, sort_keys=True)
        Returns a JSON file representing the state of the object.

    to_yaml(self)
        Returns a YAML file representing the state of the object.

    from_yaml
        Loads a YAML file and updates the state of the object

    from_json
        Loads a JSON file and updates the state of the object
    """

    def save(self, filename, file_type='json'):
        if file_type == "yaml":
            txt = self.to_yaml()
        else:
            txt = self.to_json()
        with open(filename, 'w') as fp:
            fp.write(txt)

    def load(self, filename, file_type='json'):
        if file_type == "json":
            self.from_json(filename=filename)
        else:
            self.from_yaml(filename=filename)

    def to_dict(self):
        return self.__getstate__()

    def to_json(self, indent=4, sort_keys=True):
        return json.dumps(self.__getstate__(), indent=indent, sort_keys=sort_keys)

    def to_yaml(self):
        return yaml.dump(self.__getstate__())

    def from_yaml(self, string="", filename=None):
        if filename is not None:
            with open(filename, 'r') as fp:
                j = yaml.load(fp)
        elif string is not None:
            j = json.loads(string)
        else:
            j = dict()
        self.__setstate__(j)

    def from_json(self, string="", filename=None):
        """Reads the content of a JSON file into the object via the __setstate__ method.
        If a filename is not specified (None) the string is parsed.

        Parameters
        ----------

        string : str
            A string containing the JSON file

        filename: str
            The filename to be opened

        """
        state = dict()
        if filename is not None:
            with open(filename, 'r') as fp:
                state = json.load(fp)
        elif string is not None:
            state = json.loads(string)
        else:
            pass
        self.__setstate__(state)

    def __setstate__(self, state):
        self.kw = state

    def __getstate__(self):
        try:
            state = self.kw.copy()
        except AttributeError:
            state = dict()
        return state

    def __setattr__(self, k, v):
        try:
            kw = self.__dict__['kw']
        except KeyError:
            kw = dict()
            self.__dict__['kw'] = kw
        if k in kw:
            self.__dict__['kw'][k] = v

        # Set the attributes normally
        propobj = getattr(self.__class__, k, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, v)
        else:
            super(PythonBase, self).__setattr__(k, v)

    def __getattr__(self, key):
        return self.kw[key]

    def __repr__(self):
        return str(self.__getstate__())

    def __str__(self):
        s = 'class: %s\n' % self.__class__.__name__
        for key in self.kw.keys():
            s += "%s: \t %s\n" % (key, self.kw[key])
        return s

    def __init__(self, *args, **kwargs):
        super(PythonBase, self).__init__()
        name = kwargs.pop('name', self.__class__.__name__)
        kwargs['name'] = name() if callable(name) else name
        kwargs['verbose'] = kwargs.get('verbose', False)
        self.kw = kwargs


class AccessibleVolume(PythonBase):
    """AccessibleVolume class

    Examples
    --------
    >>> import numpy as np
    >>> atoms_xyz = np.zeros((3,11))
    >>> atoms_vdw = np.zeros(11)
    >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
    >>> linker_length = 20.0
    >>> linker_width = 2.0
    >>> dye_radius = 3.5
    >>> simulation_grid_spacing = 0.9
    >>> dye_attachment_point = np.zeros(3)
    >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
    >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)

    Attributes
    ----------
    labeling_site_name : string
        the name of the labeling site
    xyzr : array
        a numpy array containing the cartesian coordinates of all considered obstacles (atoms)
        including their radii.
    points_av : array
        array containing all non-zero points of the AV as cartesian coordinates, xyz,
        including the weights, w, of the points. The weights are normalized to unity.
        For instance for an AV with two grid points located at the positions [1,2,3]
        and [4,5,6] with the weights 0.1 and 0.9 the attribute points returns
        [[1,2,3, 0.1], [4,5,6, 0.9]] as an numpy array.
    points_contact_volume : array
        array containing the points of the contact volume with a non-zero density (see
        description of the attribute "points_av").
    grid_density_av : array
        3D-grid array
    density_contact_volume : array
        3D-grid array
    attachment_coordinate : array
        The cartesian coordinates of the grid used for the AV calculations.
    parameters : dict
         all parameters determining the weighting of the av density
         contact_volume_thickness
         contact_volume_trapped_fraction

    """

    @property
    def grid_shape(self):
        """The number of grid-points in each direction
        """
        return self.density_av.shape

    @property
    def grid_minimum_linker_length(self):
        """The minimum length of a linker to reach a certain point on a grid
        considering the obstacles (atoms). All reachable grid points are bigger
        than zero.
        """
        if self._min_linker_length is None:
            self._min_linker_length = AccessibleVolume.calculate_min_linker_length(
                self.xyzr,
                self.attachment_coordinate,
                self.parameters
            )
        return self._min_linker_length

    @property
    def grid_density_av(self):
        """A three-dimensional array of the accessible volume. The values of the array are bigger than one
        if the dye can reach the point.
        """
        if self._density_av is None:
            self.update_av()
        return self._density_av

    @property
    def points_av(self):
        """The cartesian coordinates of all points with a positive density
        """
        if self._density_av is None:
            self.update_av()
        return self._points_av

    @property
    def Rmp(self):
        """The mean position of the accessible volume (average x, y, z coordinate)

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3) + 1.0
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_1.Rmp
        array([1.01010383, 1.01010383, 1.01010383])
        """
        points = self.points_av
        weights = points[:, 3]
        sw = np.sum(weights)
        xm = np.dot(points[:, 0], weights)
        ym = np.dot(points[:, 1], weights)
        zm = np.dot(points[:, 2], weights)
        return np.array([xm, ym, zm], dtype=np.float64) / sw

    def dRmp(self, av):
        """Calculate the distance between the mean positions with respect to the accessible volume `av`

        Remark: The distance between the mean positions is not a real distance

        :param av: accessible volume object
        :return:

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.dRmp(av_2)
        17.410537005358634
        """
        return av_functions.dRmp(self, av)

    def dRDA(self, av, **kwargs):
        """Calculate the mean distance to a second AccessibleVolume object

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.dRDA(av_2)
        25.816984253459424
        >>> av_1.dRDA(av_2, distance_sampling_method="sobol_sequence", distance_samples=100)
        """
        kwargs['grid1'] = self.grid_density_av
        kwargs['grid2'] = av.grid_density_av
        kwargs['dg_1'] = self.parameters['simulation_grid_resolution']
        kwargs['dg_2'] = av.parameters['simulation_grid_resolution']
        kwargs['grid_origin_1'] = self.attachment_coordinate
        kwargs['grid_origin_2'] = av.attachment_coordinate
        return av_functions.average_distance(
            self.points_av,
            av.points_av,
            **kwargs
        )

    def mean_fret_efficiency(self, av, forster_radius=50, **kwargs):
        """

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.mean_fret_efficiency(av_2)
        0.9421520823306685
        """
        kwargs['grid1'] = self.grid_density_av
        kwargs['grid2'] = av.grid_density_av
        kwargs['dg_1'] = self.parameters['simulation_grid_resolution']
        kwargs['dg_2'] = av.parameters['simulation_grid_resolution']
        kwargs['grid_origin_1'] = self.attachment_coordinate
        kwargs['grid_origin_2'] = av.attachment_coordinate

        return av_functions.mean_fret_efficiency(
            self.points_av,
            av.points_av,
            forster_radius,
            **kwargs
        )

    def dRDAE(self, av, forster_radius=50.0, **kwargs):
        """Calculate the FRET-averaged mean distance to a second accessible volume

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.dRDAE(av_2)
        31.492959923454578
        """
        mean_e = self.mean_fret_efficiency(av, forster_radius, **kwargs)
        return (1. / mean_e - 1.) ** (1. / 6.) * forster_radius

    def pRDA(self, av, **kwargs):
        """Histogram of the inter-AV distance distribution

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> count, rda = av_1.pRDA(av_2)

        """
        kwargs['grid1'] = self.grid_density_av
        kwargs['grid2'] = av.grid_density_av
        kwargs['dg_1'] = self.parameters['simulation_grid_resolution']
        kwargs['dg_2'] = av.parameters['simulation_grid_resolution']
        kwargs['grid_origin_1'] = self.attachment_coordinate
        kwargs['grid_origin_2'] = av.attachment_coordinate

        return av_functions.histogram_rda(self.points_av, av.points_av, **kwargs)

    def distance(self, av, distance_type='RDAMeanE', **kwargs):
        """Calculates the distance of one

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.distance(av_2, distance_type="RDAMean")
        25.772462770891366
        >>> av_1.distance(av_2, distance_type="RDAMeanE")
        31.39394405317073
        >>> av_1.distance(av_2, distance_type="Efficiency", forster_radius=60)
        """
        if distance_type == "RDAMeanE":
            return self.dRDAE(av, **kwargs)
        elif distance_type == "RDAMean":
            return self.dRDA(av)
        elif distance_type == "Efficiency":
            return self.mean_fret_efficiency(av, **kwargs)
        elif distance_type == "Rmp":
            return self.dRmp(av)
        else:
            return self.dRmp(av)

    def save_av(self, write_dir, mode='xyz', **kwargs):
        """Saves the accessible volume as xyz-file or open-dx density file

        Parameters
        ----------
        write_dir : string
            The directory where the AV file is saved.
        filename : string
            The filename to which the AV is saved. By default the filename of an AV is
            the labeling_site_name attribute of the AccessibleVolume object.
        mode : string
            The

        Examples
        --------

        """
        filename = kwargs.get('av_filename', self.parameters['labeling_site_name'])
        fn = os.path.join(write_dir, filename + '.'+mode)
        openDX = kwargs.get('openDX', None)
        if openDX is not None:
            mode = "dx" if openDX else ""

        if mode == 'dx':
            density = kwargs.get('density', self.density)
            d = density / density.max() * 0.5
            ng = self.ng
            dg = self.dg
            offset = (ng - 1) / 2 * dg
            av_functions.write_open_dx(
                fn,
                d,
                self.x0 - offset,
                ng, ng, ng,
                dg, dg, dg
            )
        else:
            p = kwargs.get('points', self.points)
            d = p[:, [3]].flatten()
            d /= max(d) * 50.0
            xyz = p[:, [0, 1, 2]]
            av_functions.write_points(
                filename=fn,
                points=xyz,
                mode=mode,
                verbose=self.verbose,
                density=d
            )

    def update_av(self):
        """Recalculates the AV
        """
        xyzr = self.xyzr
        attachment_coordinate = self.attachment_coordinate
        parameters = self.parameters
        points, density_3d, attachment_coordinate = AccessibleVolume.calculate_av(
            xyzr,
            attachment_coordinate,
            parameters
        )
        self._points_av = points
        self._density_av = density_3d

    @staticmethod
    def calculate_av(xyzr, attachment_coordinate, parameters):

        if parameters['simulation_type'] == 'AV3':
            av = ll.dyeDensityAV1(
                xyzr,
                attachment_coordinate,
                parameters['linker_length'],
                parameters['linker_width'],
                [
                    parameters['radius1'],
                    parameters['radius2'],
                    parameters['radius3']
                ],
                parameters['simulation_grid_resolution']
            )
        else:
            av = ll.dyeDensityAV1(
                xyzr,
                attachment_coordinate,
                parameters['linker_length'],
                parameters['linker_width'],
                parameters['radius1'],
                parameters['simulation_grid_resolution']
            )

        density_3d = np.array(av.grid).reshape(av.shape, order='F')
        grid_origin = av.originXYZ
        dg = av.discStep
        points = av_functions.density2points(dg, density_3d, grid_origin)
        return points, density_3d, attachment_coordinate

    @staticmethod
    def calculate_min_linker_length(xyzr, attachment_coordinate, parameters):
        """ Calculates a 3D grid filled with the minimum linker length to
        reach a certain grid point. By default AV1 calculations using the first
        radius (radius1) are used.

        Parameters
        ----------
        xyzr : array
        attachment_coordinate : array
        parameters : dict

        Returns
        -------
        array :
        """
        linker_length = parameters['linker_length']
        linker_diameter = parameters['linker_width']
        dye_radius = parameters['radius1']
        disc_step = parameters['simulation_grid_resolution']
        av = ll.minLinkerLength(
            xyzr,
            attachment_coordinate,
            linker_length,
            linker_diameter,
            dye_radius,
            disc_step
        )
        return np.array(av.grid).reshape(av.shape, order='F')

    def __init__(self,
                 xyzr,
                 attachment_coordinate,
                 linker_length, linker_width, radius1,
                 **kwargs):
        """


        Parameters
        ----------

        xyzr : numpy array
            A numpy array containing the cartesian coordinates of all obstacles including their
            radii.

        linker_length : float
            The length of the linker connecting the dye moiety.

        linker_ width : float
            The width of the linker connecting the dye moiety.

        radius1 : float
            A radius of the dye moiety. This parameter is used for the "AV1" and the "AV3" calculations.

        radius2 : float
            A radius of the dye moiety. For "AV3" calculations this parameter needs to be specified.

        radius3 : float
            A radius of the dye moiety. For "AV3" calculations this parameter needs to be specified.

        attachment_coordinate : numpy array
            The cartesian coordinates where the dye linker is attached to, e.g. np.array([1., 2.1, 3.2])

        simulation_grid_resolution : float
            The resolution of the grid used for the AV simulations (default value 0.5)

        simulation_type : string
            Parameter specifying the type the the AV simulation. Currently, "AV1" and "AV3" are supported
            for calculations of the accessible volume with one or three dye radii, respectively. For "AV1"
            calculations only the parameter "radius1" needs to be specified. For "AV3" calculations the
            parameters "radius1", "radius2", and "radius3" need to be specified.

        contact_volume_thickness : float
            Parameter defining the thickness of the contact volume. All grid points which are closer to the
            molecular van der Waals surface than "contact_volume_thickness" are considered as parts of the
            contact volume.

        contact_volume_trapped_fraction : float
            The fraction of dyes located in the contact volume.


        """
        self.xyzr = xyzr
        parameters = dict()
        self.attachment_coordinate = attachment_coordinate
        parameters['labeling_site_name'] = kwargs.get('labeling_site_name', "")
        parameters['verbose'] = kwargs.get('verbose', False)
        parameters['linker_length'] = linker_length
        parameters['linker_width'] = linker_width
        parameters['radius1'] = radius1
        parameters['radius2'] = kwargs.get('radius2', None)
        parameters['radius3'] = kwargs.get('radius3', None)
        parameters['simulation_type'] = kwargs.get('simulation_type', 'AV1')
        parameters['simulation_grid_resolution'] = kwargs.get('simulation_grid_resolution', 0.5)
        parameters['min_sphere_volume_thickness'] = kwargs.get('min_sphere_volume_thickness', None)
        parameters['contact_volume_thickness'] = kwargs.get('contact_volume_thickness', None)
        parameters['contact_volume_trapped_fraction'] = kwargs.get('contact_volume_trapped_fraction', None)
        self.parameters = parameters
        super(AccessibleVolume, self).__init__(self, **parameters)

        self._min_linker_length = None
        self._density_av = None
        self._points_av = None

    def __len__(self):
        return self.points_av.shape[0]

    def __str__(self):
        s = 'Accessible Volume\n'
        s += '----------------\n'
        s += 'n-points   : %i\n' % len(self)
        s += 'attachment : %.3f, %.3f, %.3f\n' % (
            self.attachment_coordinate[0],
            self.attachment_coordinate[1],
            self.attachment_coordinate[2]
        )
        s += "\n"
        s += super(self.__class__, self).__str__()
        return s


class AVTrajectory(PythonBase):
    """
    Examples
    --------

    >>> import mdtraj as md
    >>> import avtraj

    Load some trajectory

    >>> traj = md.load('./examples/hGBP1_out_3.h5')

    Make a new accessible volume trajectory. This places a dye on the specified atom. The dye parameters are either
    passes as or taken from a dye-library (see dye_definition.json). If no dye-parameters are passed default
    parameters are used (not recommended).

    >>> av_traj_1 = avtraj.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB')

    For visual inspection the accessible volume can be saved as xyz-file.

    >>> av_traj[0].save_xyz('test_344.xyz')#

    A preset dye-parameter is loaded using the argument `dye_parameter_set`. Here the string has to match the string in the
    dye-definition.json file.

    >>> av_traj_2 = mdtraj.fluorescence.fps.AVTrajectory(traj, '18D', attachment_atom_selection='resSeq 7 and name CB', dye_parameter_set='D3Alexa488')

    """

    def __init__(self, trajectory, position_name, dye_parameter_set=None, **kwargs):
        """
        Parameters
        ----------
             chain_identifier="", residue_seq_number=None, residue_name="", atom_name="",
             strip_mask="",
             allowed_sphere_radius=0.0,
                         anchor_atoms=None,
                                          attachmet_atom_idx=None,
                                          min_sphere_volume_thickness



        Attributes
        ----------
        :param trajectory:
        :param position_name:
        :param dye_parameter_set:
        :param kwargs:
        """
        # Trajectory
        self.verbose = kwargs.get('verbose', False)
        if isinstance(trajectory, str):
            trajectory = mdtraj.load(trajectory)
        self.trajectory = trajectory
        self._avs = dict()

        # Determine vdw-radii from topology
        self.vdw = kwargs.get('vdw', av_functions.get_vdw(trajectory))

        # AV-position
        self.position_name = position_name
        self.simulation_type = kwargs.get('simulation_type', 'AV1')
        attachment_atom_index = kwargs.get('attachment_atom_index', None)
        attachment_atom_selection = kwargs.get('attachment_atom_selection', None)
        strip_mask = kwargs.get('strip_mask', None)

        if self.verbose:
            print "Attachment atom"
            print attachment_atom_selection
            print type(attachment_atom_selection)

        # Determine attachment atom index
        if isinstance(attachment_atom_index, int):
            self.attachment_atom_index = attachment_atom_index
        elif isinstance(attachment_atom_selection, unicode) or isinstance(attachment_atom_selection, str):
            topology = trajectory.topology
            index = topology.select(attachment_atom_selection)
            if self.verbose:
                print "Using selection string to determine attachment position"
                print "Attachment atom: %s" % index
            if len(index) != 1:
                raise ValueError("The selection does not result in a single atom. Please change your selection.")
            else:
                self.attachment_atom_index = index[0]
        else:
            raise ValueError("Provide either the attachment atom index or a selection string.")

        # Determine either the AV-parameters by the preset dye-parameter
        # or use the parameters provided by the user
        if isinstance(dye_parameter_set, str):
            p = dye_definition[dye_parameter_set]
            self.linker_length = p['linker_length']
            self.linker_width = p['linker_width']
            self.radius_1 = p['radius1']
            self.radius_2 = p['radius2']
            self.radius_3 = p['radius3']
            self.allowed_sphere_radius = p['allowed_sphere_radius']
            self.simulation_grid_resolution = p['simulation_grid_resolution']
        else:
            self.linker_length = kwargs.get('linker_length', 20.0)
            self.linker_width = kwargs.get('linker_width', 0.5)
            self.radius_1 = kwargs.get('radius_1', 4.5)
            self.radius_2 = kwargs.get('radius_2', 4.5)
            self.radius_3 = kwargs.get('radius_3', 4.5)
            self.allowed_sphere_radius = kwargs.get('allowed_sphere_radius', 1.0)
            self.simulation_grid_resolution = kwargs.get('simulation_grid_resolution', 0.5)

        self.av_parameter = {
            'vdw': self.vdw,
            'l': self.linker_length,
            'w': self.linker_width,
            'r1': self.radius_1,
            'r2': self.radius_1,
            'r3': self.radius_2,
            'dg': self.simulation_grid_resolution,
            'atom_i': self.attachment_atom_index,
            'strip_mask': ''
        }
        if strip_mask:
            self.vdw[topology.select(strip_mask)] *= 0.0

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, key):
        simulation_type = self.simulation_type

        if isinstance(key, int):
            frame_idx = [key]
        else:
            start = 0 if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            frame_idx = range(start, min(stop, len(self)), step)

        re = []
        for frame_i in frame_idx:
            frame = self.trajectory[frame_i]
            x = (frame.xyz[0, :, 0] * 10.0).astype(np.float64, order='C')
            y = (frame.xyz[0, :, 1] * 10.0).astype(np.float64, order='C')
            z = (frame.xyz[0, :, 2] * 10.0).astype(np.float64, order='C')

            # if av was already calculated use pre-calculated av
            if frame_i in self._avs.keys():
                av = self._avs[frame_i]
            else:
                parameters = self.av_parameter
                if simulation_type == 'AV1':
                    points, density, x0 = calculate_1_radius(x, y, z, **parameters)
                elif simulation_type == 'AV3':
                    points, density, ng, x0 = av_functions.calculate_3_radius(x, y, z, **parameters)
                av = AccessibleVolume(points, density, x0, parameters)
                self._avs[frame_i] = av
            re.append(av)

        if len(re) == 1:
            return re[0]
        else:
            return re


class AvDistanceTrajectory(object):
    """
    The AvPotential class provides the possibility to calculate the reduced or unreduced chi2 given a set of
    labeling positions and experimental distances. Here the labeling positions and distances are provided as
    dictionaries.

    Examples
    --------

    distance_file = './examples/hGBP1_distance.json'
    av_dist = mdtraj.fluorescence.fps.AvDistanceTrajectory(traj, distance_file)
    av_dist[:3]
    """

    def __init__(self, trajectory, distance_file, **kwargs):
        d = json.load(open(distance_file, 'r'))
        self.distances = d['Distances']
        self.positions = d['Positions']
        self.n_av_samples = kwargs.get('av_samples', 10000)
        self.bins = kwargs.get('hist_bins', np.linspace(10, 100, 90))
        self.trajectory = trajectory
        self._d = dict()
        self.verbose = kwargs.get('verbose', False)
        self.vdw = av_functions.get_vdw(trajectory)

    def get_avs(self, traj_index):
        frame = self.trajectory[traj_index]
        re = OrderedDict()
        arguments = [
            dict(
                {
                    'vdw': self.vdw,
                    'trajectory': frame,
                    'position_name': position_key,
                },
                **self.positions[position_key]
            )
            for position_key in self.positions
        ]
        avs = map(lambda x: AVTrajectory(**x), arguments)
        for i, position_key in enumerate(self.positions):
            re[position_key] = avs[i]
        return re

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

        re = dict((key, {'rMP': [], 'rDA': [], 'rDAE': [], 'pRDA': [], 'chi2': []}) for key in self.distances.keys())
        for frame_i in frame_idx:
            # Don't repeat calculations
            if frame_i in self._d.keys():
                rDA, rDAE, rMP, chi2, pRDA = self._d[frame_i]
            else:
                # calculate the AVs of the frame
                avs = self.get_avs(frame_i)
                # Calculate the distances
                for distance_key in self.distances:
                    distance = self.distances[distance_key]
                    av1 = avs[distance['position1_name']][0]
                    av2 = avs[distance['position2_name']][0]
                    R0 = distance['Forster_radius']

                    # Calulate first a set of random distances
                    # Use this set for the caluclation of the FRET
                    # observables
                    ran_dist = av_functions.random_distances(av1.points, av2.points, self.n_av_samples)
                    bin_edges, pRDA = av_functions.histogram_rda(distances=ran_dist,
                                                                 bins=self.bins,
                                                                 n_samples=self.n_av_samples)
                    rDA = av_functions.average_distance(distances=ran_dist, nSamples=self.n_av_samples)
                    rDAE = av_functions.RDAMeanE(distances=ran_dist, forster_radius=R0, nSamples=self.n_av_samples)
                    rMP = av_functions.dRmp(av1, av2)
                    if self.verbose:
                        print "RDA: %s" % rDA
                        print "RDA_E: %s" % rDAE
                        print "RDA_mp: %s" % rMP
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
                self._d[frame_i] = (rDA, rDAE, rMP, chi2, list(pRDA))

            for distance_key in self.distances:
                re[distance_key]['rMP'].append(rMP)
                re[distance_key]['rDA'].append(rDA)
                re[distance_key]['rDAE'].append(rDAE)
                re[distance_key]['pRDA'].append(pRDA)
                re[distance_key]['chi2'].append(chi2)

        return re