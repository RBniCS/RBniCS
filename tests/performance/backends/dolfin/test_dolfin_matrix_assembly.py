# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import dx, FunctionSpace, grad, inner, TestFunction, TrialFunction, UnitSquareMesh
from rbnics.backends import AffineExpansionStorage
from rbnics.backends import product as factory_product, sum as factory_sum
from rbnics.backends.dolfin import product as dolfin_product, sum as dolfin_sum
from test_dolfin_utils import RandomDolfinFunction, RandomTuple

product = None
sum = None
all_product = {"dolfin": dolfin_product, "factory": factory_product}
all_sum = {"dolfin": dolfin_sum, "factory": factory_sum}


class Data(object):
    def __init__(self, Th, Q):
        self.Q = Q
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.a = lambda k: k * inner(grad(u), grad(v)) * dx

    def generate_random(self):
        a = ()
        for i in range(self.Q):
            # Generate random vector
            k = RandomDolfinFunction(self.V)
            # Generate random form
            a += (self.a(k),)
        A = AffineExpansionStorage(a)
        # Genereate random theta
        theta = RandomTuple(self.Q)
        # Return
        return (theta, A)

    def evaluate_builtin(self, theta, A):
        result_builtin = theta[0] * A[0]
        for i in range(1, self.Q):
            result_builtin += theta[i] * A[i]
        return result_builtin

    def evaluate_backend(self, theta, A):
        return sum(product(theta, A))

    def assert_backend(self, theta, A, result_backend):
        result_builtin = self.evaluate_builtin(theta, A)
        relative_error = (result_builtin - result_backend).norm("frobenius") / result_builtin.norm("frobenius")
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("Q", [10 + 4 * j for j in range(1, 4)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_product.keys()))
def test_dolfin_matrix_assembly(Th, Q, test_type, benchmark):
    data = Data(Th, Q)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()) + ", Q = " + str(Q))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global product, sum
        product, sum = all_product[test_type], all_sum[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
