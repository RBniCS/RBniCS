# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import FunctionSpace, UnitSquareMesh
from rbnics.backends import BasisFunctionsMatrix
from test_dolfin_utils import RandomDolfinFunction, RandomNumpyVector


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
        uN = RandomNumpyVector(self.N)
        # Return
        return (Z, uN)

    def evaluate_builtin(self, Z, uN):
        result_builtin = uN[0] * Z[0].vector()
        for i in range(1, self.N):
            result_builtin.add_local(uN[i] * Z[i].vector().get_local())
        result_builtin.apply("add")
        return result_builtin

    def evaluate_backend(self, Z, uN):
        return (Z * uN).vector()

    def assert_backend(self, Z, uN, result_backend):
        result_builtin = self.evaluate_builtin(Z, uN)
        relative_error = (result_builtin - result_backend).norm("l2") / result_builtin.norm("l2")
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("N", [10 + 4 * j for j in range(1, 4)])
@pytest.mark.parametrize("test_type", ["builtin", "__mul__"])
def test_dolfin_Z_uN(Th, N, test_type, benchmark):
    data = Data(Th, N)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()) + ", N = " + str(N))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
