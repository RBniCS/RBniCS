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

from .navier_stokes_tensor3_pod_galerkin_reduced_problem import NavierStokesTensor3PODGalerkinReducedProblem
from .navier_stokes_tensor3_problem import NavierStokesTensor3Problem
#from rbnics.problems.navier_stokes.navier_stokes_rb_reduced_problem import NavierStokesTensor3RBReducedProblem
from .navier_stokes_tensor3_reduced_problem import NavierStokesTensor3ReducedProblem


__all__ = [
    'NavierStokesTensor3PODGalerkinReducedProblem',
    'NavierStokesTensor3Problem',
#    'NavierStokesTensor3RBReducedProblem',
    'NavierStokesTensor3ReducedProblem'
]
