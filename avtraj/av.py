from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from . base import PythonBase
from . import av_functions


class AccessibleVolume(
    PythonBase
):
    """Container class for accessible volumes

    An AccessibleVolume represents a collection of parameters needed to
    calculate the positional distribution of a flexible label around its
    attachment point. Moreover, AccessibleVolume objects facilitate the
    calculation of distances to other AccessibleVolume objects by a set
    of predefined methods.

    Examples
    --------
    >>> import numpy as np
    >>> import avtraj as avt
    >>> atoms_xyz = np.zeros((3,11))
    >>> atoms_vdw = np.zeros(11)
    >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
    >>> linker_length = 20.0
    >>> linker_width = 2.0
    >>> dye_radius = 3.5
    >>> simulation_grid_spacing = 0.9
    >>> dye_attachment_point = np.zeros(3)
    >>> parameters = dict()
    >>> av_parameters['linker_length'] = 20.0
    >>> av_parameters['linker_width'] = 0.5
    >>> av_parameters['radius1'] = 3.5
    >>> av1 = avt.AccessibleVolume(xyzr, dye_attachment_point, **parameters)

    Attributes
    ----------
    labeling_site_name : string
        the name of the labeling site
    xyzr : array
        a numpy array containing the cartesian coordinates of all considered
        obstacles (atoms) including their radii.
    points_av : array
        array containing all non-zero points of the AV as cartesian
        coordinates, xyz, including the weights, w, of the points. The
        weights are normalized to unity. For instance for an AV with two
        grid points located at the positions [1,2,3] and [4,5,6] with the
        weights 0.1 and 0.9 the attribute points returns [[1,2,3, 0.1],
        [4,5,6, 0.9]] as an numpy array.
    points_contact_volume : array
        array containing the points of the contact volume with a non-zero
        density (see description of the attribute "points_av").
    grid_density_av : array
        3D-grid array the values <= 0.0 for inaccessible points. In AV3 mode
        1/3 if the grid point is accessible by one out of 3 radii, 2/3,
        and 3/3 if it is accessible by 2 and 3 radii, respectively.
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
    def xyzr(
            self
    ) -> np.ndarray:
        """Cartesian coordinates and radii of the obstacles
        """
        return self._xyzr

    @property
    def allowed_sphere_radius(
            self
    ) -> float:
        return self._allowed_sphere_radius

    @allowed_sphere_radius.setter
    def allowed_sphere_radius(
            self,
            v: float
    ):
        self._allowed_sphere_radius = v
        self.update_av()

    @property
    def grid_shape(
            self
    ) -> Tuple[int, int, int]:
        """The number of grid-points in each direction
        """
        return self.grid_density.shape

    @property
    def grid_minimum_linker_length(self):
        """The minimum length of a linker to reach a certain point on a grid
        considering the obstacles (atoms). All reachable grid points are bigger
        than zero.
        """
        if self._min_linker_length is None:
            self._min_linker_length = av_functions.calculate_min_linker_length(
                self.xyzr,
                self.attachment_coordinate,
                self.parameters
            )
        return self._min_linker_length

    @property
    def grid_density(self):
        """A three-dimensional array of the accessible volume. The values of
        the array are bigger than one if the dye can reach the point.
        """
        if self._grid_density is None:
            self.update_av()
        return self._grid_density

    @property
    def points(self):
        """The cartesian coordinates of all points with a positive density
        """
        if self._grid_density is None:
            self.update_av()
        return self._points

    @property
    def Rmp(self):
        """The mean position of the accessible volume (average x, y, z
        coordinate)

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3) + 1.0
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_1.Rmp
        array([1.01010383, 1.01010383, 1.01010383])
        """
        points = self.points
        weights = points[:, 3]
        sw = np.sum(weights)
        xm = np.dot(points[:, 0], weights)
        ym = np.dot(points[:, 1], weights)
        zm = np.dot(points[:, 2], weights)
        return np.array(
            [xm, ym, zm], dtype=np.float64
        ) / sw

    @property
    def interaction_sites(self):
        """An array containing the coordinates of the interaction sites with
        radii and weights. If no interaction sites were specified all atoms
        are considered
        """
        return self._xyzrq

    def dRmp(
            self,
            av: AccessibleVolume
    ):
        """Calculate the distance between the mean positions with respect to
        the accessible volume `av`

        Remark: The distance between the mean positions is not a real distance

        :param av: accessible volume object
        :return:

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = avt.AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.dRmp(av_2)
        17.410537005358634
        """
        return av_functions.dRmp(self, av)

    def widthRDA(
            self,
            av: AccessibleVolume,
            **kwargs
    ):
        """Calculates the width of the distance distribution

        :param av:
        :param kwargs:
        :return:
        """
        return av_functions.widthRDA(self, av, **kwargs)

    def dRDA(
            self,
            av: AccessibleVolume,
            **kwargs
    ):
        """Calculate the mean distance to a second AccessibleVolume object

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = avt.AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.dRDA(av_2)
        25.816984253459424
        >>> av_1.dRDA(av_2, distance_sampling_method="sobol_sequence", distance_samples=100)
        """
        #return av_functions.average_distance(self, av, **kwargs)
        return av_functions.average_distance_labellib(self, av, **kwargs)

    def mean_fret_efficiency(
            self,
            av: AccessibleVolume,
            forster_radius: float = 50.0,
            **kwargs
    ):
        """

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = avt.AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.mean_fret_efficiency(av_2)
        0.9421520823306685
        """
        #return av_functions.mean_fret_efficiency(self, av, forster_radius, **kwargs)
        return av_functions.mean_fret_efficiency_label_lib(self, av, forster_radius, **kwargs)

    def dRDAE(
            self,
            av: AccessibleVolume,
            forster_radius: float = 50.0,
            **kwargs
    ):
        """Calculate the FRET-averaged mean distance to a second accessible volume

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = avt.AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> av_1.dRDAE(av_2)
        31.492959923454578
        """
        mean_e = self.mean_fret_efficiency(av, forster_radius, **kwargs)
        return (1. / mean_e - 1.) ** (1. / 6.) * forster_radius

    def pRDA(
            self,
            av: AccessibleVolume,
            **kwargs
    ):
        """Histogram of the inter-AV distance distribution

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as atv
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = atv.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = atv.AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
        >>> count, rda = av_1.pRDA(av_2)

        """
        return av_functions.histogram_rda(self, av, **kwargs)

    def distance(
            self,
            av: AccessibleVolume,
            distance_type: str = 'RDAMeanE',
            **kwargs
    ):
        """Calculates the distance of one

        Parameters
        ----------
        av : AccessibleVolume

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_2 = avt.AccessibleVolume(xyzr, dye_attachment_point + 10.0, linker_length, linker_width, dye_radius)
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

    def save_av(
            self,
            filename: str = None,
            write_dir: str = ".",
            mode: str = 'xyz',
            **kwargs
    ):
        """Saves the accessible volume as xyz-file or open-dx density file

        Parameters
        ----------
        write_dir : string
            The directory where the AV file is saved.
        filename : string (optional)
            The filename to which the AV is saved. By default the filename of
            an AV is the labeling_site_name attribute of the AccessibleVolume
            object.
        mode : string
            'dx' saves openDX file otherwise the points are saved as xyz file.
        mode_options : string
            for 'dx' files 'linker_length' saves the path_length

        Examples
        --------
        >>> import numpy as np
        >>> import avtraj as avt
        >>> atoms_xyz = np.zeros((3,11))
        >>> atoms_vdw = np.zeros(11)
        >>> xyzr = np.vstack([atoms_xyz, atoms_vdw])
        >>> linker_length = 20.0
        >>> linker_width = 2.0
        >>> dye_radius = 3.5
        >>> simulation_grid_spacing = 0.9
        >>> dye_attachment_point = np.zeros(3)
        >>> av_1 = avt.AccessibleVolume(xyzr, dye_attachment_point, linker_length, linker_width, dye_radius)
        >>> av_1.save_av(filename='test_av')
        >>> av_1.save_av(filename='test_av', openDX=True)

        """
        if filename is None:
            filename = self.parameters['labeling_site_name']
        openDX = kwargs.get('openDX', None)
        if openDX is not None:
            mode = "dx" if openDX else ""
        fn = os.path.join(
            write_dir,
            filename + '.'+ mode
        )
        mode_options = kwargs.get('mode_options', '')

        if mode == 'dx':
            if mode_options == 'linker_length':
                density = self.grid_minimum_linker_length
            else:
                density = kwargs.get('density', self.grid_density)
            d = density / density.max() * 0.5
            ng = self.grid_shape[0]
            dg = self.parameters['simulation_grid_resolution']
            x0 = self.attachment_coordinate
            offset = (ng - 1) / 2 * dg
            av_functions.write_open_dx(
                fn,
                d,
                x0 - offset,
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
        av = self
        xyzr = av.xyzr
        attachment_coordinate = av.attachment_coordinate

        # calculate an accessible volume
        parameters = av.parameters
        av_all = av_functions.calculate_av(
            xyzr,
            attachment_coordinate,
            **parameters
        )
        density_all = np.array(av_all.grid).reshape(av_all.shape, order='F')

        if parameters['simulation_type'] == 'AV1':
            density_all = np.clip(density_all, 0, 1)

        # calculate the av of "free" dyes, i.e., the dyes which are not
        # in proximity to atoms which are considered for the contact volume.
        # To calculate the "free" AV. The atoms contributing to the ACV
        # add a weight to the density (here -5). In a later step the
        # negative numbers are replaced by zeros (masked).
        if len(av.interaction_sites) > 0:
            non_uniform_acv = self.parameters.get('non_uniform_acv', False)
            if non_uniform_acv:
                # Calculate and weight the "free" AV by subtracting the ACV
                tmp = np.copy(av.interaction_sites)
                tmp[4] *= -100
                av_free = av_functions.ll.addWeights(av_all, tmp)
                density_free = np.array(av_free.grid).reshape(av_free.shape, order='F')
                density_free = np.clip(density_free, 0, 1)

                density_acv_mask = np.clip(density_all - density_free, 0, 1)
                density_free *= (1. - av.contact_volume_trapped_fraction) / density_free.sum()

                # calculate acv
                density_acv = np.array(av_all.grid).reshape(av_all.shape, order='F')
                density_acv = np.clip(density_acv, 0, 1)

                acv = av_functions.ll.Grid3D(av_all.shape, av_all.originXYZ, av_all.discStep)
                acv.grid = list(density_acv.flatten())
                acv = av_functions.ll.addWeights(acv, av.interaction_sites)

                density_acv = np.array(acv.grid).reshape(
                    acv.shape, order='F'
                ) * density_acv_mask
                density_acv *= av.contact_volume_trapped_fraction / density_acv.sum()

                density_all = density_acv + density_free

            else:
                xyzrq = av.interaction_sites
                xyzrq[4] *= -100
                av_free = av_functions.ll.addWeights(av_all, xyzrq)
                density_free = np.array(av_free.grid).reshape(
                    av_free.shape, order='F'
                )
                density_free = np.clip(density_free, 0, 1)

                # calculate acv
                density_acv = np.clip(density_all - density_free, 0, 1)

                # weight the density of the AV and the ACV
                density_acv *= av.contact_volume_trapped_fraction / max(
                    1, density_acv.sum()
                )
                density_free *= (1. - av.contact_volume_trapped_fraction) / max(
                    1, density_free.sum()
                )

                density_all = density_acv + density_free
        else:
            density_all /= density_all.sum()

        self._points = av_all.points().T
        self._grid_density = density_all
        av_all.grid = list(density_all.flatten(order='F'))
        self._av_grid = av_all

    def __init__(
            self,
            xyzr: np.ndarray,
            attachment_coordinate: np.ndarray,
            **kwargs
    ):
        """
        Parameters
        ----------

        xyzr : numpy array
            A numpy array containing the cartesian coordinates of all obstacles
            including their radii.

        interaction_sites_xyzrq : numpy array
            A numpy array containing the cartesian coordinates, the radii, and
            the weights of the positions considered for the calculation of the
            accessible contact volume. If "xyzrq_contact" is not specified upon
            initialization of an AccessibleVolume object all positions
            specified by the parameter xyzr are with a radii defined by
            contact_volume_thickness and the radius given by xyzr are
            considered with uniform weighting for the calculation of the
            ACV.

        linker_length : float
            The length of the linker connecting the dye moiety.

        linker_width : float
            The width of the linker connecting the dye moiety.

        radius1 : float
            A radius of the dye moiety. This parameter is used for the "AV1"
            and the "AV3" calculations.

        radius2 : float
            A radius of the dye moiety. For "AV3" calculations this parameter
            needs to be specified.

        radius3 : float
            A radius of the dye moiety. For "AV3" calculations this parameter
            needs to be specified.

        attachment_coordinate : numpy array
            The cartesian coordinates where the dye linker is attached to, e.g.,
             np.array([1., 2.1, 3.2])

        simulation_grid_resolution : float
            The resolution of the grid used for the AV simulations (default
            value 0.5)

        simulation_type : string
            Parameter specifying the type the the AV simulation. Currently,
            "AV1" and "AV3" are supported for calculations of the accessible
            volume with one or three dye radii, respectively. For "AV1"
            calculations only the parameter "radius1" needs to be specified.
            For "AV3" calculations the parameters "radius1", "radius2", and
            "radius3" need to be specified.

        contact_volume_thickness : float
            Parameter defining the thickness of the contact volume. All grid
            points which are closer to the molecular van der Waals surface than
            "contact_volume_thickness" are considered as parts of the
            contact volume.

        contact_volume_trapped_fraction : float
            The fraction of dyes located in the contact volume.

        allowed_sphere_radius : float
            Excludes atoms within this radius around the attachment point.


        """
        self._xyzrq = kwargs.pop('interaction_sites_xyzrq', [])
        self._allowed_sphere_radius = kwargs.pop('allowed_sphere_radius', 0.0)
        super().__init__(self, **kwargs)

        self.parameters = kwargs
        self._xyzr = np.copy(xyzr)

        # mask atoms with distances within allowed sphere radius around attachment coordinate
        x0 = np.hstack([attachment_coordinate, 0.0])
        asr = self.allowed_sphere_radius
        sd = (xyzr.T - x0)**2
        d2 = sd[:, 0] + sd[:, 1] + sd[:, 2]
        xyzr.T[np.where(d2 < asr ** 2)[0]] *= np.array([1.0, 1.0, 1.0, 0.0])

        # calculate an interaction array if no array was provided
        #if self._xyzrq is None:
        #    xyzc = np.copy(self.xyzr)
        #    xyzc[3] += self.contact_volume_thickness
        #    w = np.ones(xyzc.shape[1], dtype=np.float64)
        #    self._xyzrq = np.vstack([xyzc, w])

        self.attachment_coordinate = attachment_coordinate
        self._min_linker_length = None
        self._grid_density = None
        self._points = None
        self._av_grid = None

        self.update_av()

    def __len__(self):
        return self.points.shape[0]

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
        s += super().__str__()
        return s