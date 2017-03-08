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
## @file __init__.py
#  @brief RBniCS: reduced order modelling in FEniCS
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

__author__ = "Francesco Ballarin, Gianluigi Rozza, Alberto Sartori"
__copyright__ = "Copyright 2015-2017 by the RBniCS authors"
__license__ = "LGPL"
__version__ = "0.0.1"
__email__ = "francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it"

# Set empty __all__ variable to be possibly extended by backends
__all__ = []

# Import the minimum subset of RBniCS required to run tutorials
from RBniCS.eim import DEIM, EIM, ExactParametrizedFunctions
from RBniCS.problems.elliptic_coercive import EllipticCoerciveProblem
from RBniCS.problems.elliptic_optimal_control import EllipticOptimalControlProblem
from RBniCS.problems.parabolic_coercive import ParabolicCoerciveProblem
from RBniCS.problems.stokes import StokesProblem
from RBniCS.problems.stokes_optimal_control import StokesOptimalControlProblem
from RBniCS.sampling import DrawFrom, EquispacedDistribution, LogUniformDistribution, UniformDistribution
from RBniCS.scm import SCM, ExactCoercivityConstant
from RBniCS.shape_parametrization import ShapeParametrization
from RBniCS.utils.decorators import exact_problem
from RBniCS.utils.factories import ReducedBasis, PODGalerkin

__all__ += [
    # RBniCS.eim
    'DEIM',
    'EIM',
    'ExactParametrizedFunctions',
    # RBniCS.problems
    'EllipticCoerciveProblem',
    'EllipticOptimalControlProblem',
    'ParabolicCoerciveProblem',
    'StokesProblem',
    'StokesOptimalControlProblem',
    # RBniCS.sampling
    'DrawFrom',
    'EquispacedDistribution',
    'LogUniformDistribution',
    'UniformDistribution',
    # RBniCS.scm
    'SCM',
    'ExactCoercivityConstant',
    # RBniCS.shape_parametrization
    'ShapeParametrization',
    # RBniCS.utils.decorators
    'exact_problem',
    # RBniCS.utils.factories
    'ReducedBasis',
    'PODGalerkin',
]
