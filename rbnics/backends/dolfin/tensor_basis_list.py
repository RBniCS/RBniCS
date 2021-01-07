# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.basic import TensorBasisList as BasicTensorBasisList
from rbnics.backends.dolfin.tensors_list import TensorsList
from rbnics.utils.decorators import BackendFor

TensorBasisList_Base = BasicTensorBasisList(TensorsList)


@BackendFor("dolfin", inputs=(FunctionSpace, ))
class TensorBasisList(TensorBasisList_Base):
    pass
