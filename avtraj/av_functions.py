import os
import json
from string import ascii_lowercase

import numpy as np
import numba as nb
from math import sqrt

import LabelLib as ll

DISTANCE_SAMPLES = 200000
DISTANCE_SAMPLING_METHOD = "random"

package_directory = os.path.dirname(os.path.abspath(__file__))
_periodic_table = json.load(open(os.path.join(package_directory, 'elements.json')))['Periodic Table']
VDW_DICT = dict((key, _periodic_table[key]["vdW radius"]) for key in _periodic_table.keys())

LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=0)}


PDB_KEYS = [
    'i', 'chain', 'res_id', 'res_name',
    'atom_id', 'atom_name', 'element',
    'coord',
    'charge', 'radius', 'bfactor', 'mass'
]

PDB_FORMATS = [
    'i4', '|S1', 'i4', '|S5',
    'i4', '|S5', '|S1',
    '3f8',
    'f8', 'f8', 'f8', 'f8'
]

sobol_sequence = []#sobol_lib.i4_sobol_generate(6, DISTANCE_SAMPLES)


def calculate_av(xyzr, attachment_coordinate, **kwargs):
    """Determines for a label defined by parameters defining the
    linker and the shape all grid points which can be reached by a linker.

    The naming of the parameters follows convention of the JSON file format.

    Parameters
    ----------
    xyzr : array
        Numpy array containing the cartesian coordinates and radii of the obstacles.

    attachment_coordinate : array
        Numpy array of the cartesian coordinates of the attachment point

    parameters : dict
        Python dictionary containing all parameters necessary for the calculation of an
        accessible volume (details see documentation of JSON input file)

    Returns
    -------
        The points of the AV with positive density, a 3D grid filled with the AV densities,
        and the attachment coordinates. All return values are numpy arrays. All coordinates
        are cartesian coordinates.
    """

    if kwargs['simulation_type'] == 'AV3':
        av = ll.dyeDensityAV3(
            xyzr,
            attachment_coordinate,
            kwargs['linker_length'],
            kwargs['linker_width'],
            [
                kwargs['radius1'],
                kwargs['radius2'],
                kwargs['radius3']
            ],
            kwargs['simulation_grid_resolution']
        )
    else:
        av = ll.dyeDensityAV1(
            xyzr,
            attachment_coordinate,
            kwargs['linker_length'],
            kwargs['linker_width'],
            kwargs['radius1'],
            kwargs['simulation_grid_resolution']
        )

    return av


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


def write(filename, atoms=None, append_model=False, append_coordinates=False, **kwargs):
    """ Writes a structured numpy array containing the PDB-info to a PDB-file

    If append_model and append_coordinates are False the file is overwritten. Otherwise the atomic-coordinates
    are appended to the existing file.


    :param filename: target-filename
    :param atoms: structured numpy array
    :param append_model: bool
        If True the atoms are appended as a new model
    :param append_coordinates:
        If True the coordinates are appended to the file

    """
    mode = 'a+' if append_model or append_coordinates else 'w+'
    fp = open(filename, mode)

    al = ["%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" %
          ("ATOM ", at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'], at['res_id'], " ",
           at['coord'][0], at['coord'][1], at['coord'][2], 0.0, at['bfactor'], at['element'], "  ")
          for at in atoms
    ]
    if append_model:
        fp.write('MODEL')
    fp.write("".join(al))
    if append_model:
        fp.write('ENDMDL')
    fp.close()


def write_xyz(filename, points, verbose=False):
    """
    Writes the points as xyz-format file. The xyz-format file can be opened and displayed for instance
    in PyMol

    :param filename: string
    :param points: array
    :param verbose: bool

    """
    if verbose:
        print("write_xyz\n")
        print("Filename: %s\n" % filename)
    fp = open(filename, 'w')
    npoints = len(points)
    fp.write('%i\n' % npoints)
    fp.write('Name\n')
    for p in points:
        fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))
    fp.close()


def write_points(filename, points, verbose=False, mode='xyz', density=None):
    if mode == 'pdb':
        atoms = np.empty(len(points), dtype={'names': PDB_KEYS, 'formats': PDB_FORMATS})
        atoms['coord'] = points
        if density is not None:
            atoms['bfactor'] = density
        write(filename, atoms, verbose=verbose)
    else:
        write_xyz(filename, points, verbose=verbose)


def open_dx(density, ro, rn, dr):
    """ Returns a open_dx string compatible with PyMOL

    :param density: 3d-grid with values (densities)
    :param ro: origin (x, y, z)
    :param rn: number of grid-points in x, y, z
    :param dr: grid-size (dx, dy, dz)
    :return: string
    """
    xo, yo, zo = ro
    xn, yn, zn = rn
    dx, dy, dz = dr
    s = ""
    s += "object 1 class gridpositions counts %i %i %i\n" % (xn, yn, zn)
    s += "origin " + str(xo) + " " + str(yo) + " " + str(zo) + "\n"
    s += "delta %s 0 0\n" % dx
    s += "delta 0 %s 0\n" % dy
    s += "delta 0 0 %s\n" % dz
    s += "object 2 class gridconnections counts %i %i %i\n" % (xn, yn, zn)
    s += "object 3 class array type double rank 0 items " + str(xn*yn*zn) + " data follows\n"
    n = 0
    for i in range(0, xn):
        for j in range(0, yn):
            for k in range(0, zn):
                s += str(density[i, j, k])
                n += 1
                if n % 3 == 0:
                    s += "\n"
                else:
                    s += " "
    s += "\nobject \"density (all) [A^-3]\" class field\n"

    return s


def write_open_dx(filename, density, r0, nx, ny, nz, dx, dy, dz):
    """Writes a density into a dx-file

    :param filename: output filename
    :param density: 3d-grid with values (densities)
    :param ro: origin (x, y, z)
    :param rx, ry, rz: number of grid-points in x, y, z
    :param dx, dy, dz: grid-size (dx, dy, dz)

    :return:
    """
    with open(filename, 'w') as fp:
        s = open_dx(density, r0, (nx, ny, nz), (dx, dy, dz))
        fp.write(s)


def get_vdw(trajectory):
    """Get a vector of the vdw-radii
    :param trajectory: mdtraj
    :return:
    """
    return np.array([_periodic_table[atom.element.symbol]['vdW radius'] for atom in trajectory.topology.atoms],
                    dtype=np.float64)


def histogram_rda(av1, av2, **kwargs):
    """Calculates the distance distribution with respect to a second accessible volume and returns the
    distance axis and the probability of the respective distance. By default the distance-axis "mfm.rda_axis"
    is taken to generate the histogram.

    Parameters
    ----------
    points_1 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    points_2 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    Examples
    --------
    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
    >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
    >>> y, x = av1.pRDA(av2)

    Returns
    -------
    tuple
        A tuple (p, rda_axis) where the first and the second element correspond to the
        weights and the histogram bins.

    """
    rda_axis = kwargs.get('rda_axis', None)
    if rda_axis is None:
        rda_axis = np.linspace(5, 150, 100)
    kwargs['grid1'] = av1.grid_density
    kwargs['grid2'] = av2.grid_density
    kwargs['dg_1'] = av1.parameters['simulation_grid_resolution']
    kwargs['dg_2'] = av2.parameters['simulation_grid_resolution']
    kwargs['grid_origin_1'] = av1.attachment_coordinate
    kwargs['grid_origin_2'] = av2.attachment_coordinate
    d = random_distances(av1.points, av2.points, **kwargs)

    p = np.histogram(d[:, 0], bins=rda_axis, weights=d[:, 1])[0]
    p = np.append(p, [0])
    return p, rda_axis


def average_distance(av1, av2, **kwargs):
    """Calculate the mean distance between two array of points

    Parameters
    ----------
    points_1 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    points_2 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    distance_samples : int
        The number of randomly picked distances for the calculation of the average distance between
        the two arrays of points

    use_weights : bool
        If use_weights is True the weights of the points are considered in the caclualtion of the
        average distance between the points.


    Returns
    -------
    float
        The distance between the two set of points

    """
    kwargs['grid1'] = av1.grid_density
    kwargs['grid2'] = av2.grid_density
    kwargs['dg_1'] = av1.parameters['simulation_grid_resolution']
    kwargs['dg_2'] = av2.parameters['simulation_grid_resolution']
    kwargs['grid_origin_1'] = av1.attachment_coordinate
    kwargs['grid_origin_2'] = av2.attachment_coordinate
    d = random_distances(av1.points, av2.points, **kwargs)

    if kwargs.get('use_weights', True):
        return np.dot(d[:, 0], d[:, 1]) / d[:, 1].sum()
    else:
        return np.mean(d[:, 0])


def average_distance_labellib(av1, av2, **kwargs):
    """Calculate the mean distance between two array of points

    Parameters
    ----------
    points_1 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    points_2 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    distance_samples : int
        The number of randomly picked distances for the calculation of the average distance between
        the two arrays of points

    use_weights : bool
        If use_weights is True the weights of the points are considered in the caclualtion of the
        average distance between the points.


    Returns
    -------
    float
        The distance between the two set of points

    """
    distance_samples = kwargs.get('distance_samples', DISTANCE_SAMPLES)
    return ll.meanDistance(av1._av_grid, av2._av_grid, distance_samples)


def widthRDA(av1, av2, **kwargs):
    """Calculate the width of the distance distribution between two accessible volumes

    >>> pdb_filename = '/examples/T4L_Topology.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av1 = mfm.fluorescence.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fluorescence.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fluorescence.fps.functions.widthRDA(av1, av2)
    52.93390285282142
    """
    n_samples = kwargs.get('distance_samples', DISTANCE_SAMPLES)
    d = random_distances(av1.points, av2.points, n_samples)
    s = np.dot(d[:, 0]**2.0, d[:, 1])
    f = np.dot(d[:, 0], d[:, 1])**2.0
    v = s - f
    return np.sqrt(v)


def mean_fret_efficiency(av1, av2, forster_radius=52.0, **kwargs):
    """Calculate the FRET-averaged (PDA/Intensity) distance between two accessible volumes

    Parameters
    ----------

    points_1 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    points_2 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].


    Returns
    -------
    float
        The mean FRET efficiency for the two sets of points

    Examples
    --------
    >>> pdb_filename = '/examples/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    kwargs['grid1'] = av1.grid_density
    kwargs['grid2'] = av2.grid_density
    kwargs['dg_1'] = av1.parameters['simulation_grid_resolution']
    kwargs['dg_2'] = av2.parameters['simulation_grid_resolution']
    kwargs['grid_origin_1'] = av1.attachment_coordinate
    kwargs['grid_origin_2'] = av2.attachment_coordinate

    d = random_distances(av1.points, av2.points, **kwargs)
    r = d[:, 0]
    w = d[:, 1]
    e = (1. / (1. + (r / forster_radius) ** 6.0))
    mean_fret = np.dot(w, e) / w.sum()
    return mean_fret


def mean_fret_efficiency_label_lib(av1, av2, forster_radius=52.0, **kwargs):
    """Calculate the FRET-averaged (PDA/Intensity) distance between two accessible volumes

    Parameters
    ----------

    points_1 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].

    points_2 : array
        An array of points containing the cartesian coordinates of the points and the weight of each
        point [x, y, z, weight].


    Returns
    -------
    float
        The mean FRET efficiency for the two sets of points

    Examples
    --------
    >>> pdb_filename = '/examples/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    distance_samples = kwargs.get('distance_samples', DISTANCE_SAMPLES)
    return ll.meanEfficiency(av1._av_grid, av2._av_grid, forster_radius, distance_samples)


def dRmp(av1, av2):
    """Calculate the distance between the mean position of two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/examples/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.dRmp(av1, av2)
    49.724995634807691
    """
    return np.sqrt(((av1.Rmp-av2.Rmp)**2).sum())


@nb.jit
def random_distances(p1, p2, grid1=None, grid2=None,
                     dg_1=1.0, dg_2=1.0, grid_origin_1=None,
                     grid_origin_2=None, distance_samples=None,
                     distance_sampling_method=None):
    """Calculates random cartesian distances between two set of points where every
    point has a weight

    Parameters
    ----------

    p1 : array
        Numpy array of cartesian coordinates xyz with weights. The array [[1,2,3,0.3], [4,5,6,0.7]]
        corresponds to two points with the coordinates [1,2,3] and [4,5,6] and weights of 0.3 and
        0.7, respectively.

    p2 : array
        Numpy array of cartesian coordinates xyz and weights (see description of p1).

    n_samples : int
        Number of random distances to be calculated

    Returns
    -------
    array
        Two dimensional array of distances with respective weight of the distance. The weights
        are calculated by the product of the weights of the individual points corresponding to
        the distance. The weights are normalized to unity (the sum of all weights is one).

    """

    n_p1 = p1.shape[0]
    n_p2 = p2.shape[0]

    if distance_samples is None:
        distance_samples = DISTANCE_SAMPLES
    if distance_sampling_method is None:
        distance_sampling_method = DISTANCE_SAMPLING_METHOD

    distances = np.empty((distance_samples, 2), dtype=np.float64)

    if distance_sampling_method == "sobol_sequence":
        n_distances = 0

        ng1 = grid1.shape[0]
        ng2 = grid2.shape[0]

        x0_1 = grid_origin_1[0]
        y0_1 = grid_origin_1[1]
        z0_1 = grid_origin_1[2]

        x0_2 = grid_origin_2[0]
        y0_2 = grid_origin_2[1]
        z0_2 = grid_origin_2[2]

        min_i = min(len(sobol_sequence), distance_samples)

        for i in range(min_i - 1):
            s = sobol_sequence[i]
            gp1_x = int(s[0] * ng1)
            gp1_y = int(s[1] * ng1)
            gp1_z = int(s[2] * ng1)

            gp2_x = int(s[3] * ng2)
            gp2_y = int(s[4] * ng2)
            gp2_z = int(s[5] * ng2)
            i += 1

            v1 = grid1[gp1_x, gp1_y, gp1_z]
            v2 = grid2[gp2_x, gp2_y, gp2_z]
            if v1 > 0 and v2 > 0:

                p1x = gp1_x * dg_1 + x0_1
                p1y = gp1_y * dg_1 + y0_1
                p1z = gp1_z * dg_1 + z0_1

                p2x = gp2_x * dg_2 + x0_2
                p2y = gp2_y * dg_2 + y0_2
                p2z = gp2_z * dg_2 + z0_2

                distances[n_distances, 0] = sqrt(
                    (p1x - p2x) ** 2.0 +
                    (p1y - p2y) ** 2.0 +
                    (p1z - p2z) ** 2.0
                )
                distances[n_distances, 1] = v1 * v2
                n_distances += 1
        distances = distances[:n_distances]
    else:
        for i in range(distance_samples):
            i1 = np.random.randint(0, n_p1)
            i2 = np.random.randint(0, n_p2)
            distances[i, 0] = sqrt(
                (p1[i1, 0] - p2[i2, 0]) ** 2.0 +
                (p1[i1, 1] - p2[i2, 1]) ** 2.0 +
                (p1[i1, 2] - p2[i2, 2]) ** 2.0
            )
            distances[i, 1] = p1[i1, 3] * p2[i2, 3]

    distances[:, 1] /= distances[:, 1].sum()

    return distances


@nb.jit
def density2points(dg, density_3d, grid_origin):
    nx, ny, nz = density_3d.shape
    n_max = nx * ny * nz
    points = np.empty((n_max, 4), dtype=np.float64, order='C')

    gdx = np.arange(0, nx, dtype=np.float64) * dg
    gdy = np.arange(0, ny, dtype=np.float64) * dg
    gdz = np.arange(0, nz, dtype=np.float64) * dg

    x0 = grid_origin[0]
    y0 = grid_origin[1]
    z0 = grid_origin[2]

    n = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                d = density_3d[ix, iy, iz]
                if d > 0:
                    points[n, 0] = gdx[ix] + x0
                    points[n, 1] = gdy[iy] + y0
                    points[n, 2] = gdz[iz] + z0
                    points[n, 3] = d
                    n += 1
    return points[:n]


