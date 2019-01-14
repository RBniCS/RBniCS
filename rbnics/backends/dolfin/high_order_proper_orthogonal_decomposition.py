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

from dolfin import FunctionSpace
from rbnics.backends.abstract import HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
from rbnics.backends.dolfin.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.dolfin.tensor_basis_list import TensorBasisList
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
HighOrderProperOrthogonalDecomposition_Base = BasicHighOrderProperOrthogonalDecomposition(backend, wrapping, online_backend, online_wrapping, AbstractHighOrderProperOrthogonalDecomposition, TensorSnapshotsList, TensorBasisList)

@BackendFor("dolfin", inputs=(FunctionSpace, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecomposition_Base):
    def __init__(self, V, empty_tensor):
        HighOrderProperOrthogonalDecomposition_Base.__init__(self, V, None, empty_tensor)
        
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
