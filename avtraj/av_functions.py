import ctypes as C
import platform
import os
import json
import numpy as np
import LabelLib as ll
import numba as nb
from math import sqrt

DISTANCE_SAMPLES = 5000

b, o = platform.architecture()
package_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_directory, './dll')
_periodic_table = json.load(open(os.path.join(package_directory, 'elements.json')))['Periodic Table']
VDW_DICT = dict((key, _periodic_table[key]["vdW radius"])
                for key in _periodic_table.keys())


def get_vdw(trajectory):
    """Get a vector of the vdw-radii
    :param trajectory: mdtraj
    :return:
    """
    return np.array([_periodic_table[atom.element.symbol]['vdW radius'] for atom in trajectory.topology.atoms],
                    dtype=np.float64)


def histogram_rda(av1=None, av2=None, **kwargs):
    """Calculates the distance distribution with respect to a second accessible volume and returns the
    distance axis and the probability of the respective distance. By default the distance-axis "mfm.rda_axis"
    is taken to generate the histogram.

    :param av1: Accessible volume
    :param av2: Accessible volume
    :param kwargs:
    :return:

    Examples
    --------

    >>> structure = mfm.structure.Structure('./sample_data/modelling/pdb_files/hGBP1_closed.pdb')
    >>> av1 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=18, atom_name='CB')
    >>> av2 = mfm.fluorescence.fps.BasicAV(structure, residue_seq_number=577, atom_name='CB')
    >>> y, x = av1.pRDA(av2)

    """
    rda_axis = kwargs.get('rda_axis', None)
    if rda_axis is None:
        rda_axis = np.linspace(5, 150, 100)
    same_size = kwargs.get('same_size', True)
    n_samples = kwargs.get('distance_samples', DISTANCE_SAMPLES)
    if av1 is None or av2 is None:
        ds = kwargs.get('distances', None)
    else:
        ds = random_distances(av1.points, av2.points, n_samples)
    r = ds[:, 0]
    w = ds[:, 1]
    p = np.histogram(r, bins=rda_axis, weights=w)[0]
    if same_size:
        p = np.append(p, [0])
    return p, rda_axis


def RDAMean(av1=None, av2=None, **kwargs):
    """Calculate the mean distance between two accessible volumes

    >>> pdb_filename = '/examples/T4L_Topology.pdb'
    >>> structure = mfm.structure.Structure(pdb_filename)
    >>> av1 = mfm.fluorescence.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fluorescence.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fluorescence.fps.functions.RDAMean(av1, av2)
    52.93390285282142
    """
    n_samples = kwargs.get('distance_samples', DISTANCE_SAMPLES)
    d = kwargs.get('distances', None)
    if d is None:
        d = random_distances(av1.points, av2.points, n_samples)
    return np.dot(d[:, 0], d[:, 1]) / d[:, 1].sum()


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


def RDAMeanE(av1=None, av2=None, forster_radius=52.0, **kwargs):
    """Calculate the FRET-averaged (PDA/Intensity) distance between two accessible volumes

    >>> pdb_filename = '/examples/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    n_samples = kwargs.get('distance_samples', DISTANCE_SAMPLES)
    d = kwargs.get('distances', None)
    if d is None:
        d = random_distances(av1.points, av2.points, n_samples)
    r = d[:, 0]
    w = d[:, 1]
    e = (1./(1.+(r/forster_radius)**6.0))
    mean_fret = np.dot(w, e) / w.sum()
    return (1./mean_fret - 1.)**(1./6.) * forster_radius


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
def random_distances(p1, p2, n_samples):
    """

    :param xyzw: a 4-dim vector xyz and the weight of the coordinate
    :param nSamples:
    :return:
    """

    n_p1 = p1.shape[0]
    n_p2 = p2.shape[0]

    distances = np.empty((n_samples, 2), dtype=np.float64)

    for i in range(n_samples):
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
    points = np.empty((n_max, 3), dtype=np.float64, order='C')

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
                    n += 1
    return points[:n]


def calculate_1_radius(x, y, z, vdw, l, w, r1, atom_i, dg=0.5, **kwargs):
    """
    :param l: float
        linker length
    :param w: float
        linker width
    :param r: float
        dye-radius
    :param atom_i: int
        attachment-atom index
    :param x: array
        Cartesian coordinates of atoms (x) in angstrom
    :param y: array
        Cartesian coordinates of atoms (y) in angstrom
    :param z: array
        Cartesian coordinates of atoms (z) in angstrom
    :param vdw:
        Van der Waals radii (same length as number of atoms)
    :param linkersphere: float
        Initial linker-sphere to start search of allowed dye positions
    :param linknodes: int
        By default 3
    :param vdwRMax: float
        Maximal Van der Waals radius
    :param dg: float
        Resolution of accessible volume in Angstrom
    :param verbose: bool
        If true informative output is printed on std-out


    """
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0], dtype=np.float64)
    vdw = vdw.astype(np.float64, order='C')

    atoms = np.array([x, y, z, vdw])
    dye_attachment_point = r0
    linker_length = l
    linker_width = w
    dye_radius = r1
    simulation_grid_spacing = dg
    av1 = ll.dyeDensityAV1(atoms,
                           dye_attachment_point,
                           linker_length,
                           linker_width,
                           dye_radius,
                           simulation_grid_spacing)
    density_3d = np.array(av1.grid).reshape(av1.shape, order='F')
    grid_origin = av1.originXYZ
    dg = av1.discStep

    points = density2points(dg, density_3d, grid_origin)
    return points, density_3d, r0

