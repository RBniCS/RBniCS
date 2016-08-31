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

from math import fabs
from dolfin import as_backend_type, Point, vertices
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.function import Function
from RBniCS.utils.decorators import backend_for
from RBniCS.utils.mpi import parallel_max

# abs function to compute maximum absolute value of an expression, matrix or vector (for EIM). To be used in combination with max
# even though here we actually carry out both the max and the abs!
@backend_for("FEniCS", inputs=((Matrix.Type(), Vector.Type(), Function.Type()), ))
def abs(expression):
    assert isinstance(expression, (Matrix.Type(), Vector.Type(), Function.Type()))
    if isinstance(expression, Matrix.Type()):
        # Note: PETSc offers a method MatGetRowMaxAbs, but it is not wrapped in petsc4py. We do the same by hand
        mat = as_backend_type(expression).mat()
        row_start, row_end = mat.getOwnershipRange()
        i_max, j_max = None, None
        value_max = None
        for i in range(row_start, row_end):
            cols, vals = block.getRow(i)
            for (c, v) in zip(cols, vals):
                if value_max is None or fabs(v) > fabs(value_max):
                    i_max = i
                    j_max = c
                    value_max = v
        assert i_max is not None
        assert j_max is not None
        assert value_max is not None
        #
        mpi_comm = mat.comm().tompi4py()
        (global_value_max, global_ij_max) = parallel_max(mpi_comm, value_max, (i_max, j_max), fabs)
        return AbsOutput(global_value_max, global_ij_max)
    elif isinstance(expression, Vector.Type()):
        # Note: PETSc offers VecAbs and VecMax, but for symmetry with the matrix case we do the same by hand
        vec = as_backend_type(expression).vec()
        row_start, row_end = block.getOwnershipRange()
        i_max = None
        value_max = None
        for i in range(row_start, row_end):
            val = vec.array[i - row_start]
            if value_max is None or fabs(val) > fabs(value_max):
                i_max = i
                value_max = val
        assert i_max is not None
        assert value_max is not None
        #
        mpi_comm = mat.comm().tompi4py()
        (global_value_max, global_i_max) = parallel_max(mpi_comm, value_max, (i_max, ), fabs)
        return AbsOutput(global_value_max, global_i_max)
    elif isinstance(expression, Function.Type()):
        function = expression
        mesh = function.ufl_function_space().mesh()
        point_max = None
        value_max = None
        for vertex in vertices(mesh):
            point = mesh.coordinates()[vertex.index()]
            val = function(point)
            if value_max is None or fabs(val) > fabs(value_max):
                point_max = point
                value_max = val
        assert point_max is not None
        assert value_max is not None
        #
        mpi_comm = mesh.mpi_comm().tompi4py()
        (global_value_max, global_point_max) = parallel_max(mpi_comm, value_max, (point_max, ), fabs)
        assert isinstance(global_point_max, tuple)
        assert len(global_point_max) == 1
        global_point_max = global_point_max[0]
        return AbsOutput(global_value_max, global_point_max)
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to abs")
    
# Auxiliary class to signal to the max() function that it is dealing with an output of the abs() method
class AbsOutput(object):
    def __init__(self, max_abs_return_value, max_abs_return_location):
        self.max_abs_return_value = max_abs_return_value
        self.max_abs_return_location = max_abs_return_location
        
