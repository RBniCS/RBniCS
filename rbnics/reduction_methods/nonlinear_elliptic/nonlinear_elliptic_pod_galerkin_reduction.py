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

from rbnics.utils.decorators import Extends, override, ReductionMethodFor
from rbnics.reduction_methods.base import NonlinearPODGalerkinReduction
from rbnics.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoercivePODGalerkinReduction
from rbnics.reduction_methods.nonlinear_elliptic.nonlinear_elliptic_reduction_method import NonlinearEllipticReductionMethod

NonlinearEllipticPODGalerkinReduction_Base = NonlinearEllipticReductionMethod(NonlinearPODGalerkinReduction(EllipticCoercivePODGalerkinReduction))

@Extends(NonlinearEllipticPODGalerkinReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(NonlinearEllipticProblem, "PODGalerkin")
class NonlinearEllipticPODGalerkinReduction(NonlinearEllipticPODGalerkinReduction_Base):
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        NonlinearEllipticPODGalerkinReduction_Base.__init__(self, truth_problem, **kwargs)
    
