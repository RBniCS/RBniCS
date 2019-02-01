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

import hashlib
from numbers import Number
from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from rbnics.eim.utils.decorators import add_to_map_from_parametrized_operator_to_problem, get_problem_from_parametrized_operator
from rbnics.utils.decorators import get_problem_from_solution, get_problem_from_solution_dot, overload

def ParametrizedTensorFactory(backend, wrapping):
    class _ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
        def __init__(self, form, spaces, assemble_empty_snapshot):
            AbstractParametrizedTensorFactory.__init__(self, form)
            self._form = form
            self._spaces = spaces
            self._assemble_empty_snapshot = assemble_empty_snapshot
            self._empty_snapshot = None
            self._name = None
            self._description = None
            
        def __eq__(self, other):
            return (
                isinstance(other, type(self))
                    and
                self._form == other._form
                    and
                self._spaces == other._spaces
            )
            
        def __hash__(self):
            return hash((self._form, self._spaces))
        
        def create_interpolation_locations_container(self, **kwargs):
            # Populate auxiliary_problems_and_components
            visited = set()
            auxiliary_problems_and_components = set() # of (problem, component)
            for node in wrapping.form_iterator(self._form, "nodes"):
                if node in visited:
                    continue
                # ... problem solutions related to nonlinear terms
                elif wrapping.is_problem_solution_type(node):
                    if wrapping.is_problem_solution(node):
                        (preprocessed_node, component, truth_solution) = wrapping.solution_identify_component(node)
                        truth_problem = get_problem_from_solution(truth_solution)
                        auxiliary_problems_and_components.add((truth_problem, component))
                    elif wrapping.is_problem_solution_dot(node):
                        (preprocessed_node, component, truth_solution_dot) = wrapping.solution_dot_identify_component(node)
                        truth_problem = get_problem_from_solution_dot(truth_solution_dot)
                        auxiliary_problems_and_components.add((truth_problem, component))
                    else:
                        (preprocessed_node, component, auxiliary_problem) = wrapping.get_auxiliary_problem_for_non_parametrized_function(node)
                        auxiliary_problems_and_components.add((auxiliary_problem, component))
                    # Make sure to skip any parent solution related to this one
                    visited.add(node)
                    visited.add(preprocessed_node)
                    for parent_node in wrapping.solution_iterator(preprocessed_node):
                        visited.add(parent_node)
            if len(auxiliary_problems_and_components) == 0:
                auxiliary_problems_and_components = None
            # Create reduced mesh
            assert "auxiliary_problems_and_components" not in kwargs
            kwargs["auxiliary_problems_and_components"] = auxiliary_problems_and_components
            return backend.ReducedMesh(self._spaces, **kwargs)
            
        def create_snapshots_container(self):
            return backend.TensorSnapshotsList(self._spaces, self.create_empty_snapshot())
            
        def create_empty_snapshot(self):
            if self._empty_snapshot is None:
                self._empty_snapshot = self._assemble_empty_snapshot()
            return backend.copy(self._empty_snapshot)
            
        def create_basis_container(self):
            return backend.TensorBasisList(self._spaces, self.create_empty_snapshot())
            
        def create_POD_container(self):
            return backend.HighOrderProperOrthogonalDecomposition(self._spaces, self.create_empty_snapshot())
            
        def name(self):
            if self._name is None:
                self._name = wrapping.form_name(self._form)
            return self._name
            
        def description(self):
            if self._description is None:
                self._description = PrettyTuple(self._form, wrapping.form_description(self._form), self.name())
            return self._description
            
        def is_parametrized(self):
            return wrapping.is_parametrized(self._form, wrapping.form_iterator) or self.is_time_dependent()
            
        def is_time_dependent(self):
            return wrapping.is_time_dependent(self._form, wrapping.form_iterator)
            
        @overload(lambda cls: cls)
        def __add__(self, other):
            form_sum = self._form + other._form
            output = _ParametrizedTensorFactory.__new__(type(self), form_sum)
            output.__init__(form_sum)
            # Set corresponding problem
            problems = [get_problem_from_parametrized_operator(operator) for operator in (self, other)]
            assert all([problem is problems[0] for problem in problems])
            add_to_map_from_parametrized_operator_to_problem(output, problems[0])
            # Automatically compute name starting from names of addends
            output._name = _ParametrizedTensorFactory._hash_name(self.name() + " + " + other.name())
            # This method is only used by exact parametrized operator evaluations, and not by DEIM.
            # Thus, description (which is called by DEIM during the offline phase) must never be used,
            # and the code should give an error if it is used by mistake.
            del output._description
            # Return
            return output
            
        @overload(Number)
        def __add__(self, other):
            from rbnics.backends import evaluate
            assert len(self._spaces) == 0
            return evaluate(self) + other
            
        @overload(Number)
        def __radd__(self, other):
            return self + other
            
        @overload((lambda cls: cls, Number))
        def __sub__(self, other):
            return self + (- other)
        
        @overload(backend.Function.Type())
        def __mul__(self, other):
            form_mul = self._form*other
            output = _ParametrizedTensorFactory.__new__(type(self), form_mul)
            output.__init__(form_mul)
            # Set corresponding problem
            add_to_map_from_parametrized_operator_to_problem(output, get_problem_from_parametrized_operator(self))
            # Do not recompute name, as the computed name would:
            # * account that other is a solution (or its derivative) while called from an high fidelity solve, because of
            #   known mapping from truth solution to truth problem.
            # * not be able account that other is the high fidelity representation of a reduced solution, because mapping
            #   from (high fidelity representation of) reduced solution to truth problem is not known, as a new
            #   reduced solution (and corresponding representation) is generated at every reduced solve
            # Simply re-use the existing name with a custom suffix.
            output._name = _ParametrizedTensorFactory._hash_name(self.name() + "_mul_function")
            # This method is only used by exact parametrized operator evaluations, and not by DEIM.
            # Thus, description (which is called by DEIM during the offline phase) must never be used,
            # and the code should give an error if it is used by mistake.
            del output._description
            # Return
            return output
            
        @overload(Number)
        def __mul__(self, other):
            return other*self
            
        @overload(Number)
        def __rmul__(self, other):
            form_mul = other*self._form
            output = _ParametrizedTensorFactory.__new__(type(self), form_mul)
            output.__init__(form_mul)
            # Set corresponding problem
            add_to_map_from_parametrized_operator_to_problem(output, get_problem_from_parametrized_operator(self))
            # Automatically compute name starting from current name.
            output._name = _ParametrizedTensorFactory._hash_name(str(other) + " * " + self.name())
            # This method is only used by exact parametrized operator evaluations, and not by DEIM.
            # Thus, description (which is called by DEIM during the offline phase) must never be used,
            # and the code should give an error if it is used by mistake.
            del output._description
            # Return
            return output
            
        def __neg__(self):
            return -1.*self
            
        @staticmethod
        def _hash_name(string):
            return hashlib.sha1(string.encode("utf-8")).hexdigest()
        
    return _ParametrizedTensorFactory
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.items()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
