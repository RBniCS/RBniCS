# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import zeros
from rbnics.backends.online.basic import Vector as BasicVector
from rbnics.backends.online.numpy.wrapping import Slicer
from rbnics.utils.decorators import backend_for, ModuleWrapper, OnlineSizeType


def VectorBaseType(N):
    return zeros(N)


backend = ModuleWrapper()
wrapping = ModuleWrapper(Slicer=Slicer)
_Vector_Type_Base = BasicVector(backend, wrapping, VectorBaseType)


class _Vector_Type(_Vector_Type_Base):
    def __getitem__(self, key):
        if isinstance(key, int):
            return float(_Vector_Type_Base.__getitem__(self, key))  # convert from numpy numbers wrappers
        else:
            return _Vector_Type_Base.__getitem__(self, key)

    def __iter__(self):
        return map(float, self.content.flat)

    def __array__(self, dtype=None):
        return self.content.__array__(dtype)


@backend_for("numpy", inputs=(OnlineSizeType, ))
def Vector(N):
    return _Vector_Type(N)


# Attach a Type() function
def Type():
    return _Vector_Type


Vector.Type = Type
