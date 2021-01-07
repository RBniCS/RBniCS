# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.basic import SnapshotsMatrix as BasicSnapshotsMatrix
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.utils.decorators import BackendFor

SnapshotsMatrix_Base = BasicSnapshotsMatrix(FunctionsList)


@BackendFor("dolfin", inputs=(FunctionSpace, (str, None)))
class SnapshotsMatrix(SnapshotsMatrix_Base):
    pass
