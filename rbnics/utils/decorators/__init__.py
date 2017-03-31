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

from rbnics.utils.decorators.abstract_backend import AbstractBackend, abstract_backend, abstract_online_backend, abstractmethod, abstractonlinemethod, abstractproperty
from rbnics.utils.decorators.backend_for import array_of, BackendFor, backend_for, ComputeThetaType, dict_of, DictOfThetaType, list_of, OnlineSizeType, OverrideBackendFor, override_backend_for, SameBackendFor, same_backend_for, ThetaType, tuple_of
from rbnics.utils.decorators.customize_reduction_method_for import CustomizeReductionMethodFor
from rbnics.utils.decorators.customize_reduced_problem_for import CustomizeReducedProblemFor
from rbnics.utils.decorators.dual_problem import DualProblem
from rbnics.utils.decorators.dual_reduced_problem import DualReducedProblem
from rbnics.utils.decorators.extends import Extends
from rbnics.utils.decorators.exact_problem import exact_problem
from rbnics.utils.decorators.multi_level_reduced_problem import MultiLevelReducedProblem
from rbnics.utils.decorators.multi_level_reduction_method import MultiLevelReductionMethod
from rbnics.utils.decorators.override import override
from rbnics.utils.decorators.primal_dual_reduced_problem import PrimalDualReducedProblem
from rbnics.utils.decorators.primal_dual_reduction_method import PrimalDualReductionMethod
from rbnics.utils.decorators.problem_decorator_for import ProblemDecoratorFor
from rbnics.utils.decorators.reduced_problem_decorator_for import ReducedProblemDecoratorFor
from rbnics.utils.decorators.reduced_problem_for import ReducedProblemFor
from rbnics.utils.decorators.reduction_method_decorator_for import ReductionMethodDecoratorFor
from rbnics.utils.decorators.reduction_method_for import ReductionMethodFor
from rbnics.utils.decorators.regenerate_reduced_problem_from_exact_reduced_problem import regenerate_reduced_problem_from_exact_reduced_problem
from rbnics.utils.decorators.store_map_from_problem_name_to_problem import add_to_map_from_problem_name_to_problem, get_problem_from_problem_name, StoreMapFromProblemNameToProblem
from rbnics.utils.decorators.store_map_from_problem_to_reduced_problem import add_to_map_from_problem_to_reduced_problem, get_reduced_problem_from_problem, StoreMapFromProblemToReducedProblem
from rbnics.utils.decorators.store_and_update_map_from_problem_to_training_status import is_training_finished, set_map_from_problem_to_training_status_on, set_map_from_problem_to_training_status_off, StoreMapFromProblemToTrainingStatus, UpdateMapFromProblemToTrainingStatus
from rbnics.utils.decorators.store_map_from_solution_to_problem import add_to_map_from_solution_to_problem, get_problem_from_solution, is_problem_solution, StoreMapFromSolutionToProblem
from rbnics.utils.decorators.store_problem_decorators_for_factories import StoreProblemDecoratorsForFactories
from rbnics.utils.decorators.sync_setters import sync_setters

__all__ = [
    'AbstractBackend',
    'abstract_backend',
    'abstract_online_backend',
    'abstractmethod',
    'abstractonlinemethod',
    'abstractproperty',
    'add_to_map_from_problem_name_to_problem',
    'add_to_map_from_problem_to_reduced_problem',
    'add_to_map_from_solution_to_problem',
    'array_of',
    'BackendFor',
    'backend_for',
    'ComputeThetaType',
    'CustomizeReducedProblemFor',
    'CustomizeReductionMethodFor',
    'dict_of',
    'DictOfThetaType',
    'DualProblem',
    'DualReducedProblem',
    'exact_problem',
    'get_problem_from_problem_name',
    'get_problem_from_solution',
    'get_reduced_problem_from_problem',
    'is_problem_solution',
    'is_training_finished',
    'list_of',
    'MultiLevelReducedProblem',
    'MultiLevelReductionMethod',
    'OnlineSizeType',
    'override',
    'OverrideBackendFor',
    'override_backend_for',
    'PrimalDualReducedProblem',
    'PrimalDualReductionMethod',
    'ProblemDecoratorFor',
    'ReducedProblemDecoratorFor',
    'ReducedProblemFor',
    'ReductionMethodDecoratorFor',
    'ReductionMethodFor',
    'regenerate_reduced_problem_from_exact_reduced_problem',
    'SameBackendFor',
    'same_backend_for',
    'set_map_from_problem_to_training_status_on',
    'set_map_from_problem_to_training_status_off',
    'StoreMapFromProblemNameToProblem',
    'StoreMapFromProblemToReducedProblem',
    'StoreMapFromProblemToTrainingStatus',
    'StoreMapFromSolutionToProblem',
    'StoreProblemDecoratorsForFactories',
    'sync_setters',
    'ThetaType',
    'tuple_of',
    'UpdateMapFromProblemToTrainingStatus'
]
