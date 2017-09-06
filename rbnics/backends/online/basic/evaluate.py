# Copyright (C) 2015-2017 by the RBniCS authors
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

# Evaluate a parametrized expression, possibly at a specific location
def evaluate(expression, at, backend):
    assert isinstance(expression, (backend.Matrix.Type(), backend.Vector.Type()))
    assert isinstance(at, tuple) or at is None
    for at_i in at:
        assert isinstance(at_i, (int, tuple))
        if isinstance(at_i, tuple):
            assert all([isinstance(at_ij, int) for at_ij in at_i])
            assert isinstance(expression, backend.Matrix.Type())
        else:
            assert isinstance(expression, backend.Vector.Type())
    if isinstance(expression, (backend.Matrix.Type(), backend.Vector.Type())):
        if at is None:
            return expression
        else:
            return expression[at]
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to evaluate")
    
