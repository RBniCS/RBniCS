# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic import NonAffineExpansionStorage as BasicNonAffineExpansionStorage
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import BackendFor, tuple_of

NonAffineExpansionStorage_Base = BasicNonAffineExpansionStorage


@BackendFor("numpy", inputs=((int, tuple_of(Matrix.Type()), tuple_of(Vector.Type())), (int, None)))
class NonAffineExpansionStorage(NonAffineExpansionStorage_Base):
    pass
