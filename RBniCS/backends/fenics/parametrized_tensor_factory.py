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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import Argument, assemble, Expression, Function, FunctionSpace
from RBniCS.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from RBniCS.backends.fenics.tensor_snapshots_list import TensorSnapshotsList
from RBniCS.backends.fenics.tensor_basis_list import TensorBasisList
from RBniCS.backends.fenics.wrapping import function_from_subfunction_if_any, get_form_argument, get_form_description, get_form_name
from RBniCS.utils.decorators import BackendFor, Extends, get_problem_from_solution, override, tuple_of

@Extends(AbstractParametrizedTensorFactory)
@BackendFor("fenics", inputs=(object, Form)) # object will actually be a ParametrizedDifferentialProblem
class ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
    # This are needed for proper I/O in tensor_load/tensor_save
    _all_forms = dict()
    _all_forms_assembled_containers = dict()
    
    def __init__(self, truth_problem, form):
        AbstractParametrizedTensorFactory.__init__(self, truth_problem, form)
        # Store input
        self._truth_problem = truth_problem
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
        assembled_form = assemble(form)
        assembled_form.generator = form
        ParametrizedTensorFactory._all_forms[self._name] = form
        ParametrizedTensorFactory._all_forms_assembled_containers[self._name] = assembled_form
    
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
        return TensorSnapshotsList(self._spaces)
        
    @override
    def create_basis_container(self):
        return TensorBasisList(self._spaces)
        
    @override
    def create_POD_container(self):
        return HighOrderProperOrthogonalDecomposition(self._spaces)
        
    @override
    def name(self):
        return get_form_name(self._form)
        
    @override
    def description(self):
        return PrettyTuple(self._form, get_form_description(self._form), self._name)
        
    @override
    def is_parametrized(self):
        for integral in self._form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    node = function_from_subfunction_if_any(node)
                    # ... parametrized expressions
                    if isinstance(node, Expression) and "mu_0" in node.user_parameters:
                        return True
                    # ... problem solutions related to nonlinear terms
                    elif isinstance(node, Function):
                        truth_problem = get_problem_from_solution(node)
                        return True
        return False
        
    @override
    def is_nonlinear(self):
        visited = list()
        all_truth_problems = list()
        
        for integral in self._form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    node = function_from_subfunction_if_any(node)
                    if node in visited:
                        continue
                    # ... problem solutions related to nonlinear terms
                    elif isinstance(node, Function):
                        truth_problem = get_problem_from_solution(node)
                        all_truth_problems.append(truth_problem)
                        visited.append(node)
                        
        return self._truth_problem in all_truth_problems
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.iteritems()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
        
