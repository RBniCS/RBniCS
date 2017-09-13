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
from dolfin import assemble
from rbnics.backends.basic import ParametrizedTensorFactory as BasicParametrizedTensorFactory
from rbnics.backends.dolfin.copy import copy
from rbnics.backends.dolfin.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from rbnics.backends.dolfin.reduced_mesh import ReducedMesh
from rbnics.backends.dolfin.tensor_basis_list import TensorBasisList
from rbnics.backends.dolfin.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.dolfin.wrapping import form_argument_space, form_description, form_iterator, form_name, is_parametrized, is_time_dependent
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(copy, HighOrderProperOrthogonalDecomposition, ReducedMesh, TensorBasisList, TensorSnapshotsList)
wrapping = ModuleWrapper(form_iterator, form_description=form_description, form_name=form_name, is_parametrized=is_parametrized, is_time_dependent=is_time_dependent)
ParametrizedTensorFactory_Base = BasicParametrizedTensorFactory(backend, wrapping)

@BackendFor("dolfin", inputs=(Form, ))
class ParametrizedTensorFactory(ParametrizedTensorFactory_Base):
    def __init__(self, form):
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
        empty_snapshot = assemble(form)
        empty_snapshot.zero()
        empty_snapshot.generator = self
        # Call Parent
        ParametrizedTensorFactory_Base.__init__(self, form, spaces, empty_snapshot)
    
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
                    
