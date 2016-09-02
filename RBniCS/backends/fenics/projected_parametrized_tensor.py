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
from RBniCS.backends.abstract import ProjectedParametrizedTensor as AbstractProjectedParametrizedTensor
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractProjectedParametrizedTensor)
@BackendFor("FEniCS", inputs=(Form, ReducedMesh))
class ProjectedParametrizedTensor(AbstractProjectedParametrizedTensor):
    def __init__(self, tensor, reduced_mesh):
        AbstractProjectedParametrizedTensor.__init__(tensor, reduced_mesh)
        #
        self._tensor = tensor
        self._reduced_mesh = reduced_mesh
    
    @override
    @property
    def tensor(self):
        return self._tensor
        
    @override
    @property
    def reduced_mesh(self):
        return self._reduced_mesh
        
    @override
    def get_processor_id(self, indices):
        return # TODO
        
