# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic import transpose as basic_transpose
from rbnics.utils.decorators import overload


def transpose(backend, wrapping, online_backend, online_wrapping,
              AdditionalIsFunction=None, ConvertAdditionalFunctionTypes=None,
              AdditionalIsVector=None, ConvertAdditionalVectorTypes=None,
              AdditionalIsMatrix=None, ConvertAdditionalMatrixTypes=None):

    basic_transpose_instance = basic_transpose(backend, wrapping, online_backend, online_wrapping,
                                               AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                                               AdditionalIsVector, ConvertAdditionalVectorTypes,
                                               AdditionalIsMatrix, ConvertAdditionalMatrixTypes)

    # Define a functor first, for symmetry with basic implementation
    class _Transpose_Functor(object):
        def __call__(self, arg):
            return _Transpose_Class(arg)

    # Define the actual class which will carry out the multiplication
    class _Transpose_Class(object):
        @overload(wrapping.DelayedTransposeWithArithmetic, )
        def __init__(self, arg):
            self.basic_transpose_instance_call = basic_transpose_instance(arg.evaluate())

        @overload(object, )
        def __init__(self, arg):
            self.basic_transpose_instance_call = basic_transpose_instance(arg)

        @overload(wrapping.DelayedTransposeWithArithmetic, )
        def __mul__(self, other):
            return self.basic_transpose_instance_call * other.evaluate()

        @overload(object, )
        def __mul__(self, other):
            return self.basic_transpose_instance_call * other

    return _Transpose_Functor()
