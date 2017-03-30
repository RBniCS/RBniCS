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
## @file __init__.py
#  @brief Init file for auxiliary problems module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.elliptic_coercive.elliptic_coercive_pod_galerkin_reduced_problem import EllipticCoercivePODGalerkinReducedProblem
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem_dual import EllipticCoerciveProblem_Dual
from RBniCS.problems.elliptic_coercive.elliptic_coercive_rb_non_compliant_reduced_problem import EllipticCoerciveRBNonCompliantReducedProblem
from RBniCS.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from RBniCS.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem_dual import EllipticCoerciveRBReducedProblem_Dual
from RBniCS.problems.elliptic_coercive.elliptic_coercive_reduced_problem import EllipticCoerciveReducedProblem


__all__ = [
    'EllipticCoercivePODGalerkinReducedProblem',
    'EllipticCoerciveProblem',
    'EllipticCoerciveProblem_Dual',
    'EllipticCoerciveRBNonCompliantReducedProblem',
    'EllipticCoerciveRBReducedProblem',
    'EllipticCoerciveRBReducedProblem_Dual',
    'EllipticCoerciveReducedProblem'
]
