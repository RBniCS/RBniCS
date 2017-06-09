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

from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_reduced_problem import EllipticOptimalControlReducedProblem
from rbnics.utils.decorators import Extends, ReducedProblemFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.reduction_methods.elliptic_optimal_control import EllipticOptimalControlPODGalerkinReduction

EllipticOptimalControlPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(EllipticOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem))

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(EllipticOptimalControlPODGalerkinReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticOptimalControlProblem, EllipticOptimalControlPODGalerkinReduction)
class EllipticOptimalControlPODGalerkinReducedProblem(EllipticOptimalControlPODGalerkinReducedProblem_Base):
    pass
        
