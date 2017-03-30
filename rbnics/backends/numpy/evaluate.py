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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.backends.numpy.matrix import Matrix
from rbnics.backends.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, tuple_of

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("numpy", inputs=((Matrix.Type(), Vector.Type()), (tuple_of((tuple_of(int), int)), None)))
def evaluate(expression, at=None):
    assert isinstance(expression, (Matrix.Type(), Vector.Type()))
    if isinstance(expression, (Matrix.Type(), Vector.Type())):
        if at is None:
            return expression
        else:
            return expression[at]
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to evaluate")
    
