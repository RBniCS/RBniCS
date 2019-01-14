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

from rbnics.eim.problems.deim import DEIM
from rbnics.eim.problems.deim_decorated_problem import DEIMDecoratedProblem
from rbnics.eim.problems.deim_decorated_reduced_problem import DEIMDecoratedReducedProblem
from rbnics.eim.problems.eim import EIM
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.problems.eim_decorated_problem import EIMDecoratedProblem
from rbnics.eim.problems.eim_decorated_reduced_problem import EIMDecoratedReducedProblem
from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
from rbnics.eim.problems.exact_parametrized_functions_decorated_problem import ExactParametrizedFunctionsDecoratedProblem
from rbnics.eim.problems.exact_parametrized_functions_decorated_reduced_problem import ExactParametrizedFunctionsDecoratedReducedProblem
from rbnics.eim.problems.time_dependent_eim_approximation import TimeDependentEIMApproximation

__all__ = [
    'DEIM',
    'DEIMDecoratedProblem',
    'DEIMDecoratedReducedProblem',
    'EIM',
    'EIMApproximation',
    'EIMDecoratedProblem',
    'EIMDecoratedReducedProblem',
    'ExactParametrizedFunctions',
    'ExactParametrizedFunctionsDecoratedProblem',
    'ExactParametrizedFunctionsDecoratedReducedProblem',
    'TimeDependentEIMApproximation'
]
