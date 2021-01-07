# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.basic import TensorSnapshotsList as BasicTensorSnapshotsList
from rbnics.backends.dolfin.tensors_list import TensorsList
from rbnics.utils.decorators import BackendFor

TensorSnapshotsList_Base = BasicTensorSnapshotsList(TensorsList)


@BackendFor("dolfin", inputs=(FunctionSpace, ))
class TensorSnapshotsList(TensorSnapshotsList_Base):
    pass
