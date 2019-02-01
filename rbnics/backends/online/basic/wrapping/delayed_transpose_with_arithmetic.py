# Copyright (C) 2015-2019 by the RBniCS authors
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

from numbers import Number
from rbnics.backends.abstract import BasisFunctionsMatrix as AbstractBasisFunctionsMatrix, ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from rbnics.backends.basic.wrapping import DelayedTranspose
from rbnics.utils.decorators import overload

def DelayedTransposeWithArithmetic(backend):
    class DelayedTransposeWithArithmetic_Class(object):
        def __init__(self, arg):
            assert isinstance(arg, DelayedTranspose)
            self._arg = arg
            
        @overload(lambda cls: cls)
        def __add__(self, other):
            from rbnics.backends import transpose
            self_args = self._arg._args
            other_args = other._arg._args
            assert len(self_args) in (2, 3)
            assert len(self_args) == len(other_args)
            assert isinstance(self_args[0], AbstractBasisFunctionsMatrix)
            assert isinstance(other_args[0], AbstractBasisFunctionsMatrix)
            assert self_args[0] is other_args[0]
            assert isinstance(self_args[1], AbstractParametrizedTensorFactory)
            assert isinstance(other_args[1], AbstractParametrizedTensorFactory)
            if len(self_args) == 2:
                output = transpose(self_args[0])*(self_args[1] + other_args[1])
            elif len(self_args) == 3:
                assert isinstance(self_args[2], AbstractBasisFunctionsMatrix)
                assert isinstance(other_args[2], AbstractBasisFunctionsMatrix)
                assert self_args[2] is other_args[2]
                output = transpose(self_args[0])*(self_args[1] + other_args[1])*self_args[2]
            else:
                raise ValueError("Invalid argument")
            assert isinstance(output, DelayedTranspose)
            return DelayedTransposeWithArithmetic_Class(output)
                
        @overload(lambda cls: cls)
        def __sub__(self, other):
            return self + (- other)
            
        @overload(backend.Function.Type())
        def __mul__(self, other):
            from rbnics.backends import transpose
            args = self._arg._args
            assert len(args) == 3
            assert isinstance(args[0], AbstractBasisFunctionsMatrix)
            assert isinstance(args[1], AbstractParametrizedTensorFactory)
            assert isinstance(args[2], AbstractBasisFunctionsMatrix)
            output = transpose(args[0])*(args[1]*(args[2]*other))
            assert isinstance(output, DelayedTranspose)
            return DelayedTransposeWithArithmetic_Class(output)
            
        @overload(Number)
        def __mul__(self, other):
            return other*self
            
        @overload(Number)
        def __rmul__(self, other):
            from rbnics.backends import transpose
            args = self._arg._args
            assert len(args) in (2, 3)
            assert isinstance(args[0], AbstractBasisFunctionsMatrix)
            assert isinstance(args[1], AbstractParametrizedTensorFactory)
            if len(args) == 2:
                output = transpose(args[0])*(other*args[1])
            elif len(args) == 3:
                assert isinstance(args[2], AbstractBasisFunctionsMatrix)
                output = transpose(args[0])*(other*args[1])*args[2]
            else:
                raise ValueError("Invalid argument")
            assert isinstance(output, DelayedTranspose)
            return DelayedTransposeWithArithmetic_Class(output)
        
        def __neg__(self):
            return -1.*self
            
        def evaluate(self):
            from rbnics.backends import evaluate, transpose
            args = self._arg._args
            assert len(args) in (2, 3)
            assert isinstance(args[0], AbstractBasisFunctionsMatrix)
            assert isinstance(args[1], AbstractParametrizedTensorFactory)
            if len(args) == 2:
                output = transpose(args[0])*evaluate(args[1])
            elif len(args) == 3:
                assert isinstance(args[2], AbstractBasisFunctionsMatrix)
                output = transpose(args[0])*evaluate(args[1])*args[2]
            else:
                raise ValueError("Invalid argument")
            assert not isinstance(output, DelayedTranspose)
            return output
            
    return DelayedTransposeWithArithmetic_Class
