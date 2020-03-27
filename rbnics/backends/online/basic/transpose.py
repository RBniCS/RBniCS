# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from rbnics.backends.basic import transpose as basic_transpose
from rbnics.utils.decorators import overload

def transpose(backend, wrapping, online_backend, online_wrapping, AdditionalIsFunction=None, ConvertAdditionalFunctionTypes=None, AdditionalIsVector=None, ConvertAdditionalVectorTypes=None, AdditionalIsMatrix=None, ConvertAdditionalMatrixTypes=None):

    basic_transpose_instance = basic_transpose(backend, wrapping, online_backend, online_wrapping, AdditionalIsFunction, ConvertAdditionalFunctionTypes, AdditionalIsVector, ConvertAdditionalVectorTypes, AdditionalIsMatrix, ConvertAdditionalMatrixTypes)

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
            return self.basic_transpose_instance_call*other.evaluate()

        @overload(object, )
        def __mul__(self, other):
            return self.basic_transpose_instance_call*other

    return _Transpose_Functor()
