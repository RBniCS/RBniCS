# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.abstract import (
    HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition)
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
from rbnics.backends.dolfin.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.dolfin.tensor_basis_list import TensorBasisList
from rbnics.backends.dolfin.wrapping import get_mpi_comm
from rbnics.backends.online import OnlineEigenSolver
from rbnics.utils.decorators import BackendFor, ModuleWrapper


def transpose(arg):
    # cannot import transpose at global scope due to cyclic dependence
    from rbnics.backends.dolfin.transpose import transpose as backend_transpose
    return backend_transpose(arg)


backend = ModuleWrapper(transpose)
wrapping = ModuleWrapper(get_mpi_comm)
online_backend = ModuleWrapper(OnlineEigenSolver=OnlineEigenSolver)
online_wrapping = ModuleWrapper()
HighOrderProperOrthogonalDecomposition_Base = BasicHighOrderProperOrthogonalDecomposition(
    backend, wrapping, online_backend, online_wrapping, AbstractHighOrderProperOrthogonalDecomposition,
    TensorSnapshotsList, TensorBasisList)


@BackendFor("dolfin", inputs=(FunctionSpace, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecomposition_Base):
    def __init__(self, V, empty_tensor):
        HighOrderProperOrthogonalDecomposition_Base.__init__(self, V, None, empty_tensor)

    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
