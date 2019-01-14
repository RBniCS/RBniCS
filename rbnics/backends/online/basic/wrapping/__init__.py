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

from rbnics.backends.online.basic.wrapping.delayed_transpose_with_arithmetic import DelayedTransposeWithArithmetic
from rbnics.backends.online.basic.wrapping.DirichletBC import DirichletBC
from rbnics.backends.online.basic.wrapping.function_to_vector import function_to_vector
from rbnics.backends.online.basic.wrapping.preserve_solution_attributes import preserve_solution_attributes
from rbnics.backends.online.basic.wrapping.slice_to_array import slice_to_array
from rbnics.backends.online.basic.wrapping.slice_to_size import slice_to_size

__all__ = [
    'DelayedTransposeWithArithmetic',
    'DirichletBC',
    'function_to_vector',
    'preserve_solution_attributes',
    'slice_to_array',
    'slice_to_size'
]
