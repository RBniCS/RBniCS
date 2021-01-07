# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from abc import ABCMeta, abstractmethod
from ufl.core.multiindex import MultiIndex
from ufl.indexed import Indexed
from ufl.tensors import ListTensor
from dolfin import Function
from rbnics.backends.dolfin.wrapping.get_function_space import get_function_space
from rbnics.backends.dolfin.wrapping.is_problem_solution import _split_function
from rbnics.backends.dolfin.wrapping.solution_identify_component import _remove_mute_indices
from rbnics.utils.cache import Cache, cache
from rbnics.utils.decorators import ModuleWrapper


# Only a V attribute and a name method are required
class AuxiliaryProblemForNonParametrizedFunction(object, metaclass=ABCMeta):
    def __init__(self, V):
        self.V = V

    @abstractmethod
    def name(self):
        pass


def basic_get_auxiliary_problem_for_non_parametrized_function(backend, wrapping):
    @cache  # to skip preprocessing if node is queried multiple time
    def _basic_get_auxiliary_problem_for_non_parametrized_function(node):
        assert isinstance(node, (Function, Indexed, ListTensor))
        preprocessed_node = _remove_mute_indices(node)
        assert isinstance(preprocessed_node, (Function, Indexed, ListTensor))
        if isinstance(preprocessed_node, Function):
            function = preprocessed_node
            component = (None, )
        elif isinstance(preprocessed_node, Indexed):
            assert len(preprocessed_node.ufl_operands) == 2
            assert isinstance(preprocessed_node.ufl_operands[0], Function)
            assert isinstance(preprocessed_node.ufl_operands[1], MultiIndex)
            function_split_to_component = dict()
            function_split_to_function = dict()
            _split_function(preprocessed_node.ufl_operands[0], function_split_to_component, function_split_to_function)
            assert preprocessed_node in function_split_to_component
            assert preprocessed_node in function_split_to_function
            component = function_split_to_component[preprocessed_node]
            function = function_split_to_function[preprocessed_node]
        elif isinstance(preprocessed_node, ListTensor):
            assert all(isinstance(component, Indexed) for component in preprocessed_node.ufl_operands)
            assert all(len(component.ufl_operands) == 2 for component in preprocessed_node.ufl_operands)
            assert all(isinstance(component.ufl_operands[0], Function) for component in preprocessed_node.ufl_operands)
            assert all(isinstance(component.ufl_operands[1], MultiIndex)
                       for component in preprocessed_node.ufl_operands)
            assert all(component.ufl_operands[0] == preprocessed_node.ufl_operands[-1].ufl_operands[0]
                       for component in preprocessed_node.ufl_operands)
            function_split_to_component = dict()
            function_split_to_function = dict()
            _split_function(preprocessed_node.ufl_operands[-1].ufl_operands[0], function_split_to_component,
                            function_split_to_function)
            assert preprocessed_node in function_split_to_component
            assert preprocessed_node in function_split_to_function
            component = function_split_to_component[preprocessed_node]
            function = function_split_to_function[preprocessed_node]
        else:
            raise ValueError("Invalid function provided to get_auxiliary_problem_for_non_parametrized_function")

        V = wrapping.get_function_space(function)
        try:
            auxiliary_problem_for_non_parametrized_function = _auxiliary_problem_for_non_parametrized_function_cache[V]
        except KeyError:
            # Change the name of the (local) class to uniquely identify the function.
            auxiliary_problem_for_non_parametrized_function_name = (
                AuxiliaryProblemForNonParametrizedFunction.__name__ + "_"
                + str(len(_auxiliary_problem_for_non_parametrized_function_cache))
            )

            class AuxiliaryProblemForNonParametrizedFunction_Local(AuxiliaryProblemForNonParametrizedFunction):
                def name(self):
                    return auxiliary_problem_for_non_parametrized_function_name

            AuxiliaryProblemForNonParametrizedFunction_Local.__name__ = (
                auxiliary_problem_for_non_parametrized_function_name)
            auxiliary_problem_for_non_parametrized_function = AuxiliaryProblemForNonParametrizedFunction_Local(V)
            _auxiliary_problem_for_non_parametrized_function_cache[V] = auxiliary_problem_for_non_parametrized_function

        return (preprocessed_node, component, auxiliary_problem_for_non_parametrized_function)

    _auxiliary_problem_for_non_parametrized_function_cache = Cache()
    # cache over function spaces rather than nodes, as after preprocessing all nodes sharing the same function space
    # can use the same auxiliary problem

    return _basic_get_auxiliary_problem_for_non_parametrized_function


backend = ModuleWrapper()
wrapping = ModuleWrapper(get_function_space)
get_auxiliary_problem_for_non_parametrized_function = basic_get_auxiliary_problem_for_non_parametrized_function(
    backend, wrapping)
