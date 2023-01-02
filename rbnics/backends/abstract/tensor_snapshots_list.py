# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract.tensors_list import TensorsList
from rbnics.utils.decorators import AbstractBackend


@AbstractBackend
class TensorSnapshotsList(TensorsList):
    pass
