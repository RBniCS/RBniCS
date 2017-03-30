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

from rbnics.problems.stokes import StokesPODGalerkinReducedProblem
from rbnics.problems.navier_stokes.navier_stokes_reduced_problem import NavierStokesReducedProblem
from rbnics.utils.decorators import Extends, override, ReducedProblemFor
from rbnics.problems.base import NonlinearPODGalerkinReducedProblem
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from rbnics.reduction_methods.navier_stokes import NavierStokesPODGalerkinReduction

NavierStokesPODGalerkinReducedProblem_Base = NavierStokesReducedProblem(NonlinearPODGalerkinReducedProblem(StokesPODGalerkinReducedProblem))

@Extends(NavierStokesPODGalerkinReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(NavierStokesProblem, NavierStokesPODGalerkinReduction)
class NavierStokesPODGalerkinReducedProblem(NavierStokesPODGalerkinReducedProblem_Base):
    pass
    
        
