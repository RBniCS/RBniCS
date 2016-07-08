# Copyright (C) 2015-2016 by the RBniCS authors
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
__copyright__ = "Copyright 2015-2016 by the RBniCS authors"
__license__ = "LGPL"
__version__ = "0.0.1"
__email__ = "francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it"

# Check that dolfin has been compiled with PETSc and SLEPc
from dolfin import has_petsc, has_linear_algebra_backend, parameters, has_slepc
assert has_petsc() and has_linear_algebra_backend("PETSc") and parameters.linear_algebra_backend == "PETSc"
assert has_slepc()


# Import the minimum subset of RBniCS required to run tutorials
from RBniCS.eim import ExactParametrizedFunctionEvaluation #EIM,  # TODO enable
from RBniCS.factories import ReducedBasis, PODGalerkin
from RBniCS.problems import EllipticCoerciveProblem
from RBniCS.scm import ExactCoercivityConstant # SCM, # TODO enable
from RBniCS.shape_parametrization import ShapeParametrization

__all__ = [
    # RBniCS.eim
    #'EIM', # TODO enable
    'ExactParametrizedFunctionEvaluation',
    # RBniCS.factories
    'ReducedBasis',
    'PODGalerkin',
    # RBniCS.problems
    'EllipticCoerciveProblem',
    # RBniCS.scm
    #'SCM', # TODO enable
    'ExactCoercivityConstant',
    # RBniCS.shape_parametrization
    'ShapeParametrization'
]
