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

from ufl import Form
from ufl.algorithms import expand_derivatives
from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import Argument, assemble, Expression, Function, FunctionSpace
from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from rbnics.backends.fenics.copy import copy
from rbnics.backends.fenics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from rbnics.backends.fenics.reduced_mesh import ReducedMesh
from rbnics.backends.fenics.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.fenics.tensor_basis_list import TensorBasisList
from rbnics.backends.fenics.wrapping import function_from_subfunction_if_any, get_form_argument, get_form_description, get_form_name
from rbnics.utils.decorators import BackendFor, Extends, get_problem_from_solution, is_problem_solution, override, tuple_of

@Extends(AbstractParametrizedTensorFactory)
@BackendFor("fenics", inputs=(Form, ))
class ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
    def __init__(self, form):
        AbstractParametrizedTensorFactory.__init__(self, form)
        # Store input
        form = expand_derivatives(form)
        self._form = form
        # Compute name
        self._name = get_form_name(form)
        # Extract spaces from forms
        len_spaces = len(form.arguments())
        assert len_spaces in (1, 2)
        if len_spaces == 2:
            spaces = (
                get_form_argument(form, 0).function_space(),
                get_form_argument(form, 1).function_space()
            )
        elif len_spaces == 1:
            spaces = (
                get_form_argument(form, 0).function_space(),
            )
        self._spaces = spaces
        # Store for I/O
        empty_snapshot = assemble(form)
        empty_snapshot.zero()
        empty_snapshot.generator = self
        self._empty_snapshot = empty_snapshot
    
    @override
    def create_interpolation_locations_container(self):
        # Populate subdomain data
        subdomain_data = list()
        for integral in self._form.integrals():
            if integral.subdomain_data() is not None and integral.subdomain_data() not in subdomain_data:
                subdomain_data.append(integral.subdomain_data())
        # Create reduced mesh
        if len(subdomain_data) > 0:
            reduced_mesh = ReducedMesh(self._spaces, subdomain_data)
        else:
            reduced_mesh = ReducedMesh(self._spaces)
        return reduced_mesh
        
    @override
    def create_snapshots_container(self):
        return TensorSnapshotsList(self._spaces, self._empty_snapshot)
        
    @override
    def create_empty_snapshot(self):
        snapshot = copy(self._empty_snapshot)
        return snapshot
        
    @override
    def create_basis_container(self):
        return TensorBasisList(self._spaces, self._empty_snapshot)
        
    @override
    def create_POD_container(self):
        return HighOrderProperOrthogonalDecomposition(self._spaces, self._empty_snapshot)
        
    @override
    def name(self):
        return get_form_name(self._form)
        
    @override
    def description(self):
        return PrettyTuple(self._form, get_form_description(self._form), self._name)
        
    @override
    def is_parametrized(self):
        if self.is_time_dependent():
            return True
        for integral in self._form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    node = function_from_subfunction_if_any(node)
                    # ... parametrized expressions
                    if isinstance(node, Expression) and "mu_0" in node.user_parameters:
                        return True
                    # ... problem solutions related to nonlinear terms
                    elif isinstance(node, Function) and is_problem_solution(node):
                        truth_problem = get_problem_from_solution(node)
                        return True
        return False
        
    @override
    def is_time_dependent(self):
        for integral in self._form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    node = function_from_subfunction_if_any(node)
                    # ... parametrized expressions
                    if isinstance(node, Expression) and "t" in node.user_parameters:
                        return True
                    # ... problem solutions related to nonlinear terms
                    elif isinstance(node, Function) and is_problem_solution(node):
                        truth_problem = get_problem_from_solution(node)
                        if hasattr(truth_problem, "set_time"):
                            return True
        return False
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.iteritems()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
        
