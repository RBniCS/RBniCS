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

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.abstract import HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
from rbnics.backends.online.numpy.eigen_solver import EigenSolver
from rbnics.backends.online.numpy.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.online.numpy.tensor_basis_list import TensorBasisList
from rbnics.backends.online.numpy.transpose import transpose
from rbnics.backends.online.numpy.wrapping import get_mpi_comm
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(transpose)
wrapping = ModuleWrapper(get_mpi_comm)
online_backend = ModuleWrapper(OnlineEigenSolver=EigenSolver)
online_wrapping = ModuleWrapper()
HighOrderProperOrthogonalDecomposition_Base = BasicHighOrderProperOrthogonalDecomposition(backend, wrapping, online_backend, online_wrapping, AbstractHighOrderProperOrthogonalDecomposition, TensorSnapshotsList, TensorBasisList)

@BackendFor("numpy", inputs=(AbstractFunctionsList, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecomposition_Base):
    def __init__(self, basis_functions, empty_tensor):
        HighOrderProperOrthogonalDecomposition_Base.__init__(self, basis_functions, None, empty_tensor)
        
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
