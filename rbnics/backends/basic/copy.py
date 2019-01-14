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

from rbnics.utils.decorators import list_of, overload

def copy(backend, wrapping):
    class _Copy(object):
        @overload(backend.Function.Type(), )
        def __call__(self, arg):
            return wrapping.function_copy(arg)
            
        @overload(list_of(backend.Function.Type()), )
        def __call__(self, arg):
            output = list()
            for fun in arg:
                output.append(wrapping.function_copy(fun))
            return output
            
        @overload((backend.Matrix.Type(), backend.Vector.Type()), )
        def __call__(self, arg):
            return wrapping.tensor_copy(arg)
    return _Copy()
