# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.backends.basic import TensorSnapshotsList as BasicTensorSnapshotsList
from rbnics.backends.online.numpy.tensors_list import TensorsList
from rbnics.utils.decorators import BackendFor

TensorSnapshotsList_Base = BasicTensorSnapshotsList(TensorsList)


@BackendFor("numpy", inputs=(AbstractTensorsList, ))
class TensorSnapshotsList(TensorSnapshotsList_Base):
    pass
