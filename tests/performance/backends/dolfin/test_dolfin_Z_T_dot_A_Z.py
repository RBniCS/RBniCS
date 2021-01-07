# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from numpy.linalg import norm
from dolfin import assemble, dx, grad, FunctionSpace, inner, TestFunction, TrialFunction, UnitSquareMesh
from rbnics.backends import BasisFunctionsMatrix
from rbnics.backends import transpose as factory_transpose
from rbnics.backends.dolfin import transpose as dolfin_transpose
from rbnics.backends.online.numpy import Matrix as NumpyMatrix
from test_dolfin_utils import RandomDolfinFunction

transpose = None
all_transpose = {"dolfin": dolfin_transpose, "factory": factory_transpose}


class Data(object):
    def __init__(self, Th, N):
        self.N = N
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.a = lambda k: k * inner(grad(u), grad(v)) * dx

    def generate_random(self):
        # Generate random vectors
        Z = BasisFunctionsMatrix(self.V)
        Z.init("u")
        for _ in range(self.N):
            b = RandomDolfinFunction(self.V)
            Z.enrich(b)
        k = RandomDolfinFunction(self.V)
        # Generate random matrix
        A = assemble(self.a(k))
        # Return
        return (Z, A)

    def evaluate_builtin(self, Z, A):
        result_builtin = NumpyMatrix({"u": self.N}, {"u": self.N})
        for j in range(self.N):
            A_Z_j = A * Z[j].vector()
            for i in range(self.N):
                result_builtin[i, j] = Z[i].vector().inner(A_Z_j)
        return result_builtin

    def evaluate_backend(self, Z, A):
        return transpose(Z) * A * Z

    def assert_backend(self, Z, A, result_backend):
        result_builtin = self.evaluate_builtin(Z, A)
        relative_error = norm(result_builtin - result_backend) / norm(result_builtin)
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("N", [10 + 4 * j for j in range(1, 4)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_dolfin_Z_T_dot_A_Z(Th, N, test_type, benchmark):
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
