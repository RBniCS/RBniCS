# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import FunctionSpace, UnitSquareMesh
from rbnics.backends import FunctionsList
from test_dolfin_utils import RandomDolfinFunction, RandomNumpyVector


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
        result_builtin = uN[0] * S[0].vector()
        for i in range(1, self.N):
            result_builtin.add_local(uN[i] * S[i].vector().get_local())
        result_builtin.apply("add")
        return result_builtin

    def evaluate_backend(self, S, uN):
        return (S * uN).vector()

    def assert_backend(self, S, uN, result_backend):
        result_builtin = self.evaluate_builtin(S, uN)
        relative_error = (result_builtin - result_backend).norm("l2") / result_builtin.norm("l2")
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("N", [10 + 4 * j for j in range(1, 4)])
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
