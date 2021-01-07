# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
    # cannot import transpose at global scope due to cyclic dependence
    from rbnics.backends.dolfin.transpose import transpose as backend_transpose
    return backend_transpose(arg)


backend = ModuleWrapper(transpose)
wrapping = ModuleWrapper(get_mpi_comm)
online_backend = ModuleWrapper(OnlineEigenSolver=OnlineEigenSolver)
online_wrapping = ModuleWrapper()
ProperOrthogonalDecomposition_Base = BasicProperOrthogonalDecomposition(
    backend, wrapping, online_backend, online_wrapping, AbstractProperOrthogonalDecomposition,
    SnapshotsMatrix, FunctionsList)


@BackendFor("dolfin", inputs=(FunctionSpace, (Form, Matrix.Type()), (str, None)))
class ProperOrthogonalDecomposition(ProperOrthogonalDecomposition_Base):
    def __init__(self, V, inner_product, component=None):
        ProperOrthogonalDecomposition_Base.__init__(self, V, inner_product, component)

    def store_snapshot(self, snapshot, component=None, weight=None):
        self.snapshots_matrix.enrich(snapshot, component, weight)
