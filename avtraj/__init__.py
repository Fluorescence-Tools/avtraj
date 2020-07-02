# -*- coding: utf-8 -*-
"""AvTraj - Accessible volume calculation for MD trajectories

"""
from __future__ import annotations
__version__ = "0.0.10"

import os

from . import av_functions
from .av import AccessibleVolume
from .trajectory import AvDistanceTrajectory, AVTrajectory
from .base import PythonBase

name = "avtraj"
package_directory = os.path.dirname(__file__)


