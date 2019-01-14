# Copyright (C) 2015-2019 by the RBniCS authors
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
from rbnics.problems.navier_stokes_unsteady.navier_stokes_unsteady_problem import NavierStokesUnsteadyProblem
from rbnics.problems.navier_stokes_unsteady.navier_stokes_unsteady_reduced_problem import NavierStokesUnsteadyReducedProblem
from rbnics.problems.stokes_unsteady.stokes_unsteady_pod_galerkin_reduced_problem import AbstractCFDUnsteadyPODGalerkinReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import NonlinearTimeDependentPODGalerkinReducedProblem
from rbnics.reduction_methods.navier_stokes_unsteady import NavierStokesUnsteadyPODGalerkinReduction

NavierStokesUnsteadyPODGalerkinReducedProblem_Base = AbstractCFDUnsteadyPODGalerkinReducedProblem(NonlinearTimeDependentPODGalerkinReducedProblem(NavierStokesUnsteadyReducedProblem(NavierStokesPODGalerkinReducedProblem)))

@ReducedProblemFor(NavierStokesUnsteadyProblem, NavierStokesUnsteadyPODGalerkinReduction)
class NavierStokesUnsteadyPODGalerkinReducedProblem(NavierStokesUnsteadyPODGalerkinReducedProblem_Base):
    pass
