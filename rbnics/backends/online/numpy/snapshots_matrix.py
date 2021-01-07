# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.basic import SnapshotsMatrix as BasicSnapshotsMatrix
from rbnics.backends.online.numpy.functions_list import FunctionsList
from rbnics.utils.decorators import BackendFor

SnapshotsMatrix_Base = BasicSnapshotsMatrix(FunctionsList)


@BackendFor("numpy", inputs=(AbstractFunctionsList, (str, None)))
class SnapshotsMatrix(SnapshotsMatrix_Base):
    pass
