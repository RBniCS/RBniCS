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

from ufl import Form
from dolfin import FunctionSpace
from rbnics.backends.abstract import ProperOrthogonalDecomposition as AbstractProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicProperOrthogonalDecomposition
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.snapshots_matrix import SnapshotsMatrix
from rbnics.backends.dolfin.wrapping import get_mpi_comm
from rbnics.backends.online import OnlineEigenSolver
from rbnics.utils.decorators import BackendFor, ModuleWrapper

def transpose(arg):
    from rbnics.backends.dolfin.transpose import transpose as backend_transpose # cannot import at global scope due to cyclic dependence
    return backend_transpose(arg)

backend = ModuleWrapper(transpose)
wrapping = ModuleWrapper(get_mpi_comm)
online_backend = ModuleWrapper(OnlineEigenSolver=OnlineEigenSolver)
online_wrapping = ModuleWrapper()
ProperOrthogonalDecomposition_Base = BasicProperOrthogonalDecomposition(backend, wrapping, online_backend, online_wrapping, AbstractProperOrthogonalDecomposition, SnapshotsMatrix, FunctionsList)

@BackendFor("dolfin", inputs=(FunctionSpace, (Form, Matrix.Type()), (str, None)))
class ProperOrthogonalDecomposition(ProperOrthogonalDecomposition_Base):
    def __init__(self, V, inner_product, component=None):
        ProperOrthogonalDecomposition_Base.__init__(self, V, inner_product, component)
        
    def store_snapshot(self, snapshot, component=None, weight=None):
        self.snapshots_matrix.enrich(snapshot, component, weight)
