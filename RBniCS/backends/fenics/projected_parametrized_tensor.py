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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from dolfin import assemble, FunctionSpace
from RBniCS.backends.abstract import ProjectedParametrizedTensor as AbstractProjectedParametrizedTensor
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from RBniCS.backends.fenics.tensor_snapshots_list import TensorSnapshotsList
from RBniCS.backends.fenics.tensor_basis_list import TensorBasisList
from RBniCS.backends.fenics.wrapping.get_form_name import get_form_name
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractProjectedParametrizedTensor)
@BackendFor("FEniCS", inputs=(Form, FunctionSpace))
class ProjectedParametrizedTensor(AbstractProjectedParametrizedTensor):
    # This are needed for proper I/O in tensor_load/tensor_save
    _all_forms = dict()
    _all_forms_assembled_containers = dict()
    
    def __init__(self, form, space):
        AbstractProjectedParametrizedTensor.__init__(self, form, space)
        #
        self._form = form
        self._space = space
        # Store for I/O
        form_name = get_form_name(form)
        assembled_form = assemble(form)
        assembled_form.generator = form
        ProjectedParametrizedTensor._all_forms[form_name] = form
        ProjectedParametrizedTensor._all_forms_assembled_containers[form_name] = assembled_form
    
    @override
    def create_interpolation_locations_container(self):
        subdomain_data = list()
        for integral in self._form.integrals():
            if integral.subdomain_data() is not None and integral.subdomain_data() not in subdomain_data:
                subdomain_data.append(integral.subdomain_data())
        if len(subdomain_data) > 0:
            reduced_mesh = ReducedMesh(self._space, subdomain_data)
        else:
            reduced_mesh = ReducedMesh(self._space)
        return reduced_mesh
        
    @override
    def create_snapshots_container(self):
        return TensorSnapshotsList(self._space)
        
    @override
    def create_basis_container(self):
        return TensorBasisList(self._space)
        
    @override
    def create_POD_container(self):
        return HighOrderProperOrthogonalDecomposition(self._space)
        
