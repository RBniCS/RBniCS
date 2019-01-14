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

from rbnics.utils.decorators import overload, tuple_of

def evaluate(backend):
    class _Evaluate(object):
        @overload(backend.Matrix.Type(), None)
        def __call__(self, matrix, at):
            return matrix
        
        @overload(backend.Matrix.Type(), tuple_of(int))
        def __call__(self, matrix, at):
            assert len(at) == 2
            return matrix[at]
        
        @overload(backend.Vector.Type(), None)
        def __call__(self, vector, at):
            return vector
        
        @overload(backend.Vector.Type(), tuple_of(int))
        def __call__(self, vector, at):
            assert len(at) == 1
            return vector
    return _Evaluate()
