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
#  @brief Init file for auxiliary eim module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.eim.problems.deim import DEIM
#from RBniCS.eim.problems.deim_decorated_problem import DEIMDecoratedProblem # not needed
from RBniCS.eim.problems.deim_decorated_reduced_problem import DEIMDecoratedReducedProblem
from RBniCS.eim.problems.eim import EIM
#from RBniCS.eim.problems.eim_approximation import EIMApproximation # not needed
#from RBniCS.eim.problems.eim_decorated_problem import EIMDecoratedProblem # not needed
from RBniCS.eim.problems.eim_decorated_reduced_problem import EIMDecoratedReducedProblem
from RBniCS.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
#from RBniCS.eim.problems.exact_parametrized_functions_decorated_problem import ExactParametrizedFunctionsDecoratedProblem # not needed
from RBniCS.eim.problems.exact_parametrized_functions_decorated_reduced_problem import ExactParametrizedFunctionsDecoratedReducedProblem
#from RBniCS.eim.problems.time_dependent_eim_approximation import TimeDependentEIMApproximation # not needed

__all__ = [
    'DEIM',
    'DEIMDecoratedReducedProblem',
    'EIM',
    'EIMDecoratedReducedProblem',
    'ExactParametrizedFunctions',
    'ExactParametrizedFunctionsDecoratedReducedProblem'
]