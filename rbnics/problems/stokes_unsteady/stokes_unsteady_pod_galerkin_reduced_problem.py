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

from rbnics.problems.stokes import StokesPODGalerkinReducedProblem
from rbnics.problems.stokes_unsteady.stokes_unsteady_reduced_problem import StokesUnsteadyReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.stokes_unsteady.stokes_unsteady_problem import StokesUnsteadyProblem
from rbnics.problems.base import LinearTimeDependentPODGalerkinReducedProblem
from rbnics.reduction_methods.stokes_unsteady import StokesUnsteadyPODGalerkinReduction

def AbstractCFDUnsteadyPODGalerkinReducedProblem(AbstractCFDUnsteadyPODGalerkinReducedProblem_Base):
    return AbstractCFDUnsteadyPODGalerkinReducedProblem_Base

StokesUnsteadyPODGalerkinReducedProblem_Base = AbstractCFDUnsteadyPODGalerkinReducedProblem(LinearTimeDependentPODGalerkinReducedProblem(StokesUnsteadyReducedProblem(StokesPODGalerkinReducedProblem)))

@ReducedProblemFor(StokesUnsteadyProblem, StokesUnsteadyPODGalerkinReduction)
class StokesUnsteadyPODGalerkinReducedProblem(StokesUnsteadyPODGalerkinReducedProblem_Base):
    pass
