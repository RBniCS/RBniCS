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
## @file nonlinear_elliptic_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.elliptic_coercive import EllipticCoercivePODGalerkinReducedProblem
from RBniCS.problems.nonlinear_elliptic.nonlinear_elliptic_reduced_problem import NonlinearEllipticReducedProblem
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor
from RBniCS.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from RBniCS.reduction_methods.nonlinear_elliptic import NonlinearEllipticPODGalerkinReduction

NonlinearEllipticPODGalerkinReducedProblem_Base = NonlinearEllipticReducedProblem(EllipticCoercivePODGalerkinReducedProblem)

@Extends(NonlinearEllipticPODGalerkinReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(NonlinearEllipticProblem, NonlinearEllipticPODGalerkinReduction)
class NonlinearEllipticPODGalerkinReducedProblem(NonlinearEllipticPODGalerkinReducedProblem_Base):
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        NonlinearEllipticPODGalerkinReducedProblem_Base.__init__(self, truth_problem, **kwargs)
        
