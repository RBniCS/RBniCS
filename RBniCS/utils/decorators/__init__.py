# Copyright (C) 2015-2016 by the RBniCS authors
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
#  @brief Init file for auxiliary I/O module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators.abstract_backend import AbstractBackend, abstract_backend, abstract_online_backend, abstractmethod, abstractonlinemethod, abstractproperty
from RBniCS.utils.decorators.backend_for import BackendFor, backend_for, dict_of, DictOfThetaType, list_of, OnlineSizeType, ThetaType, tuple_of
from RBniCS.utils.decorators.extends import Extends
from RBniCS.utils.decorators.exact_problem import ExactProblem
from RBniCS.utils.decorators.multi_level_reduced_problem import MultiLevelReducedProblem
from RBniCS.utils.decorators.multi_level_reduction_method import MultiLevelReductionMethod
from RBniCS.utils.decorators.override import override
from RBniCS.utils.decorators.problem_decorator_for import ProblemDecoratorFor
from RBniCS.utils.decorators.reduced_problem_decorator_for import ReducedProblemDecoratorFor
from RBniCS.utils.decorators.reduced_problem_for import ReducedProblemFor
from RBniCS.utils.decorators.reduction_method_decorator_for import ReductionMethodDecoratorFor
from RBniCS.utils.decorators.reduction_method_for import ReductionMethodFor
from RBniCS.utils.decorators.store_map_from_basis_functions_matrix_to_reduced_problem import add_to_map_from_basis_functions_matrix_to_reduced_problem, get_reduced_problem_from_basis_functions_matrix, StoreMapFromBasisFunctionsMatrixToReducedProblem
from RBniCS.utils.decorators.store_map_from_problem_to_reduced_problem import add_to_map_from_problem_to_reduced_problem, get_reduced_problem_from_problem, StoreMapFromProblemToReducedProblem
from RBniCS.utils.decorators.store_map_from_solution_to_problem import add_to_map_from_solution_to_problem, get_problem_from_solution, StoreMapFromSolutionToProblem
from RBniCS.utils.decorators.sync_setters import sync_setters

__all__ = [
    'AbstractBackend',
    'abstract_backend',
    'abstract_online_backend',
    'abstractmethod',
    'abstractonlinemethod',
    'abstractproperty',
    'add_to_map_from_basis_functions_matrix_to_reduced_problem',
    'add_to_map_from_problem_to_reduced_problem',
    'add_to_map_from_solution_to_problem',
    'BackendFor',
    'backend_for',
    'dict_of',
    'DictOfThetaType',
    'ExactProblem',
    'get_problem_from_solution',
    'get_reduced_problem_from_basis_functions_matrix',
    'get_reduced_problem_from_problem',
    'list_of',
    'MultiLevelReducedProblem',
    'MultiLevelReductionMethod',
    'OnlineSizeType',
    'override',
    'ProblemDecoratorFor',
    'ReducedProblemDecoratorFor',
    'ReducedProblemFor',
    'ReductionMethodDecoratorFor',
    'ReductionMethodFor',
    'StoreMapFromBasisFunctionsMatrixToReducedProblem',
    'StoreMapFromProblemToReducedProblem',
    'StoreMapFromSolutionToProblem',
    'sync_setters',
    'ThetaType',
    'tuple_of'
]
