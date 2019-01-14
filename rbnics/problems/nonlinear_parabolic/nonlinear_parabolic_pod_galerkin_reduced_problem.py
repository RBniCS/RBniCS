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

from rbnics.problems.nonlinear_elliptic import NonlinearEllipticPODGalerkinReducedProblem
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_reduced_problem import NonlinearParabolicReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import NonlinearTimeDependentPODGalerkinReducedProblem
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_problem import NonlinearParabolicProblem
from rbnics.reduction_methods.nonlinear_parabolic import NonlinearParabolicPODGalerkinReduction

NonlinearParabolicPODGalerkinReducedProblem_Base = NonlinearTimeDependentPODGalerkinReducedProblem(NonlinearParabolicReducedProblem(NonlinearEllipticPODGalerkinReducedProblem))

@ReducedProblemFor(NonlinearParabolicProblem, NonlinearParabolicPODGalerkinReduction)
class NonlinearParabolicPODGalerkinReducedProblem(NonlinearParabolicPODGalerkinReducedProblem_Base):
    pass
