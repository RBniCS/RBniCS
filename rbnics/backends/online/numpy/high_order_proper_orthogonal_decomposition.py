# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.abstract import (
    HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition)
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
HighOrderProperOrthogonalDecomposition_Base = BasicHighOrderProperOrthogonalDecomposition(
    backend, wrapping, online_backend, online_wrapping, AbstractHighOrderProperOrthogonalDecomposition,
    TensorSnapshotsList, TensorBasisList)


@BackendFor("numpy", inputs=(AbstractFunctionsList, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecomposition_Base):
    def __init__(self, basis_functions, empty_tensor):
        HighOrderProperOrthogonalDecomposition_Base.__init__(self, basis_functions, None, empty_tensor)

    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
