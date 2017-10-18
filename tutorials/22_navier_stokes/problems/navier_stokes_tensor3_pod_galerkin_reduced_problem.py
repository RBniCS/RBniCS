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

from rbnics.problems.navier_stokes import NavierStokesPODGalerkinReducedProblem
from .navier_stokes_tensor3_reduced_problem import NavierStokesTensor3ReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from .navier_stokes_tensor3_problem import NavierStokesTensor3Problem
from reduction_methods import NavierStokesTensor3PODGalerkinReduction

NavierStokesTensor3PODGalerkinReducedProblem_Base = NavierStokesTensor3ReducedProblem(NavierStokesPODGalerkinReducedProblem)

@ReducedProblemFor(NavierStokesTensor3Problem, NavierStokesTensor3PODGalerkinReduction)
class NavierStokesTensor3PODGalerkinReducedProblem(NavierStokesTensor3PODGalerkinReducedProblem_Base):
    pass
