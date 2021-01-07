# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.backends.basic import TensorBasisList as BasicTensorBasisList
from rbnics.backends.online.numpy.tensors_list import TensorsList
from rbnics.utils.decorators import BackendFor

TensorBasisList_Base = BasicTensorBasisList(TensorsList)


@BackendFor("numpy", inputs=(AbstractTensorsList, ))
class TensorBasisList(TensorBasisList_Base):
    pass
