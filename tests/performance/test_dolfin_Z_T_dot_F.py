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

import pytest
from numpy import isclose
from numpy.linalg import norm
from dolfin import FunctionSpace, UnitSquareMesh
from rbnics.backends import BasisFunctionsMatrix
from rbnics.backends import transpose as factory_transpose
from rbnics.backends.dolfin import transpose as dolfin_transpose
from rbnics.backends.online.numpy import Vector as NumpyVector
from test_utils import RandomDolfinFunction

transpose = None
all_transpose = {"dolfin": dolfin_transpose, "factory": factory_transpose}

class Data(object):
    def __init__(self, Th, N):
        self.N = N
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        
    def generate_random(self):
        # Generate random vectors
        Z = BasisFunctionsMatrix(self.V)
        Z.init("u")
        for _ in range(self.N):
            b = RandomDolfinFunction(self.V)
            Z.enrich(b)
        F = RandomDolfinFunction(self.V)
        # Return
        return (Z, F)
        
    def evaluate_builtin(self, Z, F):
        result_builtin = NumpyVector({"u": self.N})
        for i in range(self.N):
            result_builtin[i] = Z[i].vector().inner(F.vector())
        return result_builtin
        
    def evaluate_backend(self, Z, F):
        return transpose(Z)*F.vector()
        
    def assert_backend(self, Z, F, result_backend):
        result_builtin = self.evaluate_builtin(Z, F)
        relative_error = norm(result_builtin - result_backend)/norm(result_builtin)
        assert isclose(relative_error, 0., atol=1e-12)
        
@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("N", [10 + 4*j for j in range(1, 4)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_dolfin_Z_T_dot_F(Th, N, test_type, benchmark):
    data = Data(Th, N)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()) + ", N = " + str(N))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global transpose
        transpose = all_transpose[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
