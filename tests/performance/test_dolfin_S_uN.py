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
from dolfin import FunctionSpace, UnitSquareMesh
from rbnics.backends import FunctionsList
from test_utils import RandomDolfinFunction, RandomNumpyVector

class Data(object):
    def __init__(self, Th, N):
        self.N = N
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        
    def generate_random(self):
        # Generate random vectors
        S = FunctionsList(self.V)
        for _ in range(self.N):
            b = RandomDolfinFunction(self.V)
            S.enrich(b)
        uN = RandomNumpyVector(self.N)
        # Return
        return (S, uN)
        
    def evaluate_builtin(self, S, uN):
        result_builtin = uN[0]*S[0].vector()
        for i in range(1, self.N):
            result_builtin.add_local(uN[i]*S[i].vector().get_local())
        result_builtin.apply("add")
        return result_builtin
        
    def evaluate_backend(self, S, uN):
        return (S*uN).vector()
        
    def assert_backend(self, S, uN, result_backend):
        result_builtin = self.evaluate_builtin(S, uN)
        relative_error = (result_builtin - result_backend).norm("l2")/result_builtin.norm("l2")
        assert isclose(relative_error, 0., atol=1e-12)
        
@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("N", [10 + 4*j for j in range(1, 4)])
@pytest.mark.parametrize("test_type", ["builtin", "__mul__"])
def test_dolfin_S_uN(Th, N, test_type, benchmark):
    data = Data(Th, N)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()) + ", N = " + str(N))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
