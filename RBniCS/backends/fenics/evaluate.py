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

from dolfin import Point, project
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.projected_parametrized_tensor import ProjectedParametrizedTensor
from RBniCS.backends.fenics.projected_parametrized_expression import ProjectedParametrizedExpression
from RBniCS.utils.decorators import backend_for, tuple_of
from numpy import zeros as array, ndarray as PointType
from mpi4py.MPI import FLOAT

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("FEniCS", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), ProjectedParametrizedTensor, ProjectedParametrizedExpression), (tuple_of((tuple_of(int), int)), tuple_of((PointType, int)), None)))
def evaluate(expression_, at=None):
    assert isinstance(expression_, (Matrix.Type(), Vector.Type(), Function.Type(), ProjectedParametrizedTensor, ProjectedParametrizedExpression))
    assert at is None or isinstance(at, tuple)
    if isinstance(expression_, (Matrix.Type(), Vector.Type())):
        return # TODO
    elif isinstance(expression_, Function.Type()):
        function = expression_
        assert at is not None
        assert len(at) == 2
        point = at[0]
        process_id = at[1]
        out = None
        mpi_comm = function.ufl_function_space().mesh().mpi_comm().tompi4py()
        if mpi_comm.rank == process_id:
            out = function(point)
        out = mpi_comm.bcast(out, root=process_id)
        return out
    elif isinstance(expression_, ProjectedParametrizedTensor):
        if at is None:
            return # TODO
        else:
            return # TODO
    elif isinstance(expression_, ProjectedParametrizedExpression):
        expression = expression_.expression
        if at is None:
            space = expression_.space
            return project(expression, space)
        else:
            assert len(at) == 2
            point = at[0]
            process_id = at[1]
            out = array(expression.value_size())
            mpi_comm = expression_.space.mesh().mpi_comm().tompi4py()
            if mpi_comm.rank == process_id:
                expression.eval(out, point)
            mpi_comm.Bcast([out, FLOAT], root=process_id)
            return out
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to evaluate")
    
