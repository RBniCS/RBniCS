# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import assemble, dx, FunctionSpace, grad, inner, TestFunction, TrialFunction, UnitSquareMesh
from rbnics.backends import transpose as factory_transpose
from rbnics.backends.dolfin import transpose as dolfin_transpose
from test_dolfin_utils import RandomDolfinFunction

transpose = None
all_transpose = {"dolfin": dolfin_transpose, "factory": factory_transpose}


class Data(object):
    def __init__(self, Th):
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.a = lambda k: k * inner(grad(u), grad(v)) * dx

    def generate_random(self):
        # Generate random vectors
        v1 = RandomDolfinFunction(self.V).vector()
        v2 = RandomDolfinFunction(self.V).vector()
        k = RandomDolfinFunction(self.V)
        # Generate random matrix
        A = assemble(self.a(k))
        # Return
        return (v1, v2, A)

    def evaluate_builtin(self, v1, v2, A):
        return v1.inner(A * v2)

    def evaluate_backend(self, v1, v2, A):
        return transpose(v1) * A * v2

    def assert_backend(self, v1, v2, A, result_backend):
        result_builtin = self.evaluate_builtin(v1, v2, A)
        relative_error = (result_builtin - result_backend) / result_builtin
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(1, 9)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_dolfin_v1_dot_A_v2(Th, test_type, benchmark):
    data = Data(Th)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global transpose
        transpose = all_transpose[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
