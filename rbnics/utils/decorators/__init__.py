# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from abc import ABCMeta, abstractmethod, abstractproperty
from rbnics.utils.decorators.abstract_backend import (
    AbstractBackend, abstract_backend, abstract_online_backend, abstractonlinemethod)
from rbnics.utils.decorators.backend_for import BackendFor, backend_for
from rbnics.utils.decorators.customize_reduction_method_for import CustomizeReductionMethodFor
from rbnics.utils.decorators.customize_reduced_problem_for import CustomizeReducedProblemFor
from rbnics.utils.decorators.dispatch import (
    array_of, dict_of, dispatch, iterable_of, list_of, overload, set_of, tuple_of)
from rbnics.utils.decorators.exact_problem import exact_problem
from rbnics.utils.decorators.module_wrapper import ModuleWrapper
from rbnics.utils.decorators.online_size_type import OnlineSizeType
from rbnics.utils.decorators.parameters_type import ParametersType
from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.decorators.problem_decorator_for import ProblemDecoratorFor
from rbnics.utils.decorators.reduced_problem_decorator_for import ReducedProblemDecoratorFor
from rbnics.utils.decorators.reduced_problem_for import ReducedProblemFor
from rbnics.utils.decorators.reduction_method_decorator_for import ReductionMethodDecoratorFor
from rbnics.utils.decorators.reduction_method_for import ReductionMethodFor
from rbnics.utils.decorators.required_base_decorators import RequiredBaseDecorators
from rbnics.utils.decorators.snapshot_links_to_cache import snapshot_links_to_cache
from rbnics.utils.decorators.store_map_from_problem_name_to_problem import (
    add_to_map_from_problem_name_to_problem, get_problem_from_problem_name, StoreMapFromProblemNameToProblem)
from rbnics.utils.decorators.store_map_from_problem_to_reduced_problem import (
    add_to_map_from_problem_to_reduced_problem, get_reduced_problem_from_problem, StoreMapFromProblemToReducedProblem)
from rbnics.utils.decorators.store_map_from_problem_to_reduction_method import (
    add_to_map_from_problem_to_reduction_method, get_reduction_method_from_problem,
    StoreMapFromProblemToReductionMethod)
from rbnics.utils.decorators.store_and_update_map_from_problem_to_training_status import (
    is_training_finished, is_training_started, set_map_from_problem_to_training_status_on,
    set_map_from_problem_to_training_status_off, StoreMapFromProblemToTrainingStatus,
    UpdateMapFromProblemToTrainingStatus)
from rbnics.utils.decorators.store_map_from_solution_to_problem import (
    add_to_map_from_solution_to_problem, get_problem_from_solution, StoreMapFromSolutionToProblem)
from rbnics.utils.decorators.store_map_from_solution_dot_to_problem import (
    add_to_map_from_solution_dot_to_problem, get_problem_from_solution_dot, StoreMapFromSolutionDotToProblem)
from rbnics.utils.decorators.store_problem_decorators_for_factories import (
    StoreProblemDecoratorsForFactories)
from rbnics.utils.decorators.sync_setters import sync_setters
from rbnics.utils.decorators.theta_type import ComputeThetaType, DictOfThetaType, ThetaType

__all__ = [
    "ABCMeta",
    "AbstractBackend",
    "abstract_backend",
    "abstract_online_backend",
    "abstractmethod",
    "abstractonlinemethod",
    "abstractproperty",
    "add_to_map_from_problem_name_to_problem",
    "add_to_map_from_problem_to_reduced_problem",
    "add_to_map_from_problem_to_reduction_method",
    "add_to_map_from_solution_dot_to_problem",
    "add_to_map_from_solution_to_problem",
    "array_of",
    "BackendFor",
    "backend_for",
    "ComputeThetaType",
    "CustomizeReducedProblemFor",
    "CustomizeReductionMethodFor",
    "dict_of",
    "dispatch",
    "DictOfThetaType",
    "exact_problem",
    "get_problem_from_problem_name",
    "get_problem_from_solution",
    "get_problem_from_solution_dot",
    "get_reduced_problem_from_problem",
    "get_reduction_method_from_problem",
    "is_training_finished",
    "is_training_started",
    "iterable_of",
    "list_of",
    "ModuleWrapper",
    "OnlineSizeType",
    "overload",
    "ParametersType",
    "PreserveClassName",
    "ProblemDecoratorFor",
    "ReducedProblemDecoratorFor",
    "ReducedProblemFor",
    "ReductionMethodDecoratorFor",
    "ReductionMethodFor",
    "RequiredBaseDecorators",
    "set_map_from_problem_to_training_status_on",
    "set_map_from_problem_to_training_status_off",
    "set_of",
    "snapshot_links_to_cache",
    "StoreMapFromProblemNameToProblem",
    "StoreMapFromProblemToReducedProblem",
    "StoreMapFromProblemToReductionMethod",
    "StoreMapFromProblemToTrainingStatus",
    "StoreMapFromSolutionDotToProblem",
    "StoreMapFromSolutionToProblem",
    "StoreProblemDecoratorsForFactories",
    "sync_setters",
    "ThetaType",
    "tuple_of",
    "UpdateMapFromProblemToTrainingStatus"
]
