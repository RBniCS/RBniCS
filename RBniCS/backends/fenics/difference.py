# Copyright (C) 2015-2016 by the RBniCS authors
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

from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.wrapping import function_copy, tensor_copy
from RBniCS.utils.decorators import backend_for

# Compute the difference between two solutions
@backend_for("FEniCS", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Function.Type(), Matrix.Type(), Vector.Type())))
def difference(solution1, solution2):
    assert (
        (isinstance(solution1, Function.Type()) and isinstance(solution2, Function.Type()))
            or
        (isinstance(solution1, (Matrix.Type(), Vector.Type())) and isinstance(solution2, (Matrix.Type(), Vector.Type())))
    )
    if isinstance(solution1, Function.Type()):
        output = function_copy(solution1)
        output.vector().add_local( - solution2.vector().array() )
        output.vector().apply("add")
        return output
    elif isinstance(solution1, Matrix.Type()):
        output = tensor_copy(solution1)
        output -= solution2
        return output
    elif isinstance(solution1, Vector.Type()):
        output = tensor_copy(solution1)
        output.add_local( - solution2.array() )
        output.apply("add")
        return output
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in difference.")
    
