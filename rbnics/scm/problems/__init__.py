# Copyright (C) 2015-2018 by the RBniCS authors
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

from rbnics.scm.problems.scm import SCM
from rbnics.scm.problems.scm_approximation import SCMApproximation
from rbnics.scm.problems.scm_decorated_problem import SCMDecoratedProblem
from rbnics.scm.problems.scm_decorated_reduced_problem import SCMDecoratedReducedProblem
from rbnics.scm.problems.exact_coercivity_constant import ExactCoercivityConstant
from rbnics.scm.problems.exact_coercivity_constant_decorated_problem import ExactCoercivityConstantDecoratedProblem
from rbnics.scm.problems.exact_coercivity_constant_decorated_reduced_problem import ExactCoercivityConstantDecoratedReducedProblem
from rbnics.scm.problems.parametrized_coercivity_constant_eigenproblem import ParametrizedCoercivityConstantEigenProblem

__all__ = [
    'SCM',
    'SCMApproximation',
    'SCMDecoratedProblem',
    'SCMDecoratedReducedProblem',
    'ExactCoercivityConstant',
    'ExactCoercivityConstantDecoratedProblem',
    'ExactCoercivityConstantDecoratedReducedProblem',
    'ParametrizedCoercivityConstantEigenProblem'
]
