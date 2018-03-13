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

from ufl import Form
from ufl.algorithms import expand_derivatives
from dolfin import assemble
from rbnics.backends.basic import ParametrizedTensorFactory as BasicParametrizedTensorFactory
from rbnics.backends.dolfin.copy import copy
from rbnics.backends.dolfin.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.reduced_mesh import ReducedMesh
from rbnics.backends.dolfin.tensor_basis_list import TensorBasisList
from rbnics.backends.dolfin.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.dolfin.wrapping import form_argument_space, form_description, form_iterator, form_name, is_parametrized, is_time_dependent
from rbnics.eim.utils.decorators import add_to_map_from_parametrized_operator_to_problem, get_problem_from_parametrized_operator
from rbnics.utils.decorators import BackendFor, ModuleWrapper, overload

backend = ModuleWrapper(copy, HighOrderProperOrthogonalDecomposition, ReducedMesh, TensorBasisList, TensorSnapshotsList)
wrapping = ModuleWrapper(form_iterator, form_description=form_description, form_name=form_name, is_parametrized=is_parametrized, is_time_dependent=is_time_dependent)
ParametrizedTensorFactory_Base = BasicParametrizedTensorFactory(backend, wrapping)

@BackendFor("dolfin", inputs=(Form, ))
class ParametrizedTensorFactory(ParametrizedTensorFactory_Base):
    def __init__(self, form, assemble_empty_snapshot=True):
        # Preprocess form
        form = expand_derivatives(form)
        # Extract spaces from forms
        len_spaces = len(form.arguments())
        assert len_spaces in (1, 2)
        if len_spaces == 2:
            spaces = (
                form_argument_space(form, 0),
                form_argument_space(form, 1)
            )
        elif len_spaces == 1:
            spaces = (
                form_argument_space(form, 0),
            )
        # Create empty snapshot
        if assemble_empty_snapshot:
            empty_snapshot = assemble(form, keep_diagonal=True)
            empty_snapshot.zero()
            empty_snapshot.generator = self
            init_name_and_description = True
        else:
            empty_snapshot = None
            init_name_and_description = False
        # Call Parent
        ParametrizedTensorFactory_Base.__init__(self, form, spaces, empty_snapshot, init_name_and_description)
        
    def __eq__(self, other):
        return (
            isinstance(other, type(self))
                and
            self._form.equals(other._form)
                and
            self._spaces == other._spaces
        )
        
    def __hash__(self):
        return hash((self._form, self._spaces))
    
    def create_interpolation_locations_container(self):
        # Populate subdomain data
        subdomain_data = list()
        for integral in self._form.integrals():
            if integral.subdomain_data() is not None and integral.subdomain_data() not in subdomain_data:
                subdomain_data.append(integral.subdomain_data())
        # Create reduced mesh
        if len(subdomain_data) > 0:
            return ParametrizedTensorFactory_Base.create_interpolation_locations_container(self, subdomain_data=subdomain_data)
        else:
            return ParametrizedTensorFactory_Base.create_interpolation_locations_container(self)
            
    @overload(lambda cls: cls)
    def __add__(self, other):
        output = ParametrizedTensorFactory(self._form + other._form, False)
        problems = [get_problem_from_parametrized_operator(operator) for operator in (self, other)]
        assert all([problem is problems[0] for problem in problems])
        add_to_map_from_parametrized_operator_to_problem(output, problems[0])
        return output
        
    @overload(lambda cls: cls)
    def __sub__(self, other):
        output = ParametrizedTensorFactory(self._form - other._form, False)
        problems = [get_problem_from_parametrized_operator(operator) for operator in (self, other)]
        assert all([problem is problems[0] for problem in problems])
        add_to_map_from_parametrized_operator_to_problem(output, problems[0])
        return output
        
    @overload(Function.Type())
    def __mul__(self, other):
        output = ParametrizedTensorFactory(self._form*other, False)
        add_to_map_from_parametrized_operator_to_problem(output, get_problem_from_parametrized_operator(self))
        return output
