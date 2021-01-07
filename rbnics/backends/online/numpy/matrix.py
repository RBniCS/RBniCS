# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import zeros
from rbnics.backends.online.basic import Matrix as BasicMatrix
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import Slicer
from rbnics.utils.decorators import backend_for, ModuleWrapper, OnlineSizeType


def MatrixBaseType(M, N):
    return zeros((M, N))


backend = ModuleWrapper(Function, Vector)
wrapping = ModuleWrapper(Slicer=Slicer)
_Matrix_Type_Base = BasicMatrix(backend, wrapping, MatrixBaseType)


class _Matrix_Type(_Matrix_Type_Base):
    def __getitem__(self, key):
        if all([isinstance(key_i, int) for key_i in key]):
            return float(_Matrix_Type_Base.__getitem__(self, key))  # convert from numpy numbers wrappers
        else:
            return _Matrix_Type_Base.__getitem__(self, key)

    def __mul__(self, other):
        if isinstance(other, Vector.Type()):
            # copied from BasicMatrix because ndarray uses __matul__ instead of __mul__ for matrix-vector product
            self._arithmetic_operations_assert_attributes(other, other_order=1)
            output_content = self.content.__matmul__(other.content)
            output_size = self.M
            output = Vector.Type()(output_size, output_content)
            self._arithmetic_operations_preserve_attributes(output, other_order=1)
            return output
        else:
            return _Matrix_Type_Base.__mul__(self, other)

    def __imul__(self, other):
        if isinstance(other, Vector.Type()):
            # copied from BasicMatrix because ndarray uses __matul__ instead of __mul__ for matrix-vector product
            self._arithmetic_operations_assert_attributes(other, other_order=1)
            self.content.__imatmul__(other.content)
            return self
        else:
            return _Matrix_Type_Base.__imul__(self, other)

    def __array__(self, dtype=None):
        return self.content.__array__(dtype)


@backend_for("numpy", inputs=(OnlineSizeType, OnlineSizeType))
def Matrix(M, N):
    return _Matrix_Type(M, N)


# Attach a Type() function
def Type():
    return _Matrix_Type


Matrix.Type = Type
