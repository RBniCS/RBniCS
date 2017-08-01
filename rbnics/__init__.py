# Copyright (C) 2015-2017 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Francesco Ballarin, Gianluigi Rozza, Alberto Sartori"
__copyright__ = "Copyright 2015-2017 by the RBniCS authors"
__license__ = "LGPL"
__version__ = "0.0.1"
__email__ = "francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it"

# Set empty __all__ variable to be possibly extended by backends
__all__ = []

# Process configuration files first
from rbnics.utils.config import config

# Import the minimum subset of RBniCS required to run tutorials
from rbnics.eim import DEIM, EIM, ExactParametrizedFunctions
from rbnics.problems.elliptic_coercive import EllipticCoerciveProblem
from rbnics.problems.elliptic_optimal_control import EllipticOptimalControlProblem
from rbnics.problems.navier_stokes import NavierStokesProblem
from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
from rbnics.problems.nonlinear_parabolic import NonlinearParabolicProblem
from rbnics.problems.parabolic_coercive import ParabolicCoerciveProblem
from rbnics.problems.stokes import StokesProblem
from rbnics.problems.stokes_optimal_control import StokesOptimalControlProblem
from rbnics.problems.stokes_unsteady import StokesUnsteadyProblem
from rbnics.sampling import DrawFrom, EquispacedDistribution, LogEquispacedDistribution, LogUniformDistribution, UniformDistribution
from rbnics.scm import SCM, ExactCoercivityConstant
from rbnics.shape_parametrization import ShapeParametrization
from rbnics.utils.decorators import CustomizeReducedProblemFor, CustomizeReductionMethodFor, exact_problem
from rbnics.utils.factories import ReducedBasis, PODGalerkin

__all__ += [
    # rbnics.eim
    'DEIM',
    'EIM',
    'ExactParametrizedFunctions',
    # rbnics.problems
    'EllipticCoerciveProblem',
    'EllipticOptimalControlProblem',
    'NavierStokesProblem',
    'NonlinearEllipticProblem',
    'NonlinearParabolicProblem',
    'ParabolicCoerciveProblem',
    'StokesProblem',
    'StokesOptimalControlProblem',
    'StokesUnsteadyProblem',
    # rbnics.sampling
    'DrawFrom',
    'EquispacedDistribution',
    'LogEquispacedDistribution',
    'LogUniformDistribution',
    'UniformDistribution',
    # rbnics.scm
    'SCM',
    'ExactCoercivityConstant',
    # rbnics.shape_parametrization
    'ShapeParametrization',
    # rbnics.utils.config
    'config',
    # rbnics.utils.decorators
    'CustomizeReducedProblemFor',
    'CustomizeReductionMethodFor',
    'exact_problem',
    # rbnics.utils.factories
    'ReducedBasis',
    'PODGalerkin',
]
