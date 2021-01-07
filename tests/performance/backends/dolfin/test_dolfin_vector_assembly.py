# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import dx, FunctionSpace, TestFunction, UnitSquareMesh
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
        v = TestFunction(self.V)
        self.f = lambda g: g * v * dx

    def generate_random(self):
        f = ()
        for i in range(self.Q):
            # Generate random vector
            g = RandomDolfinFunction(self.V)
            # Generate random form
            f += (self.f(g),)
        F = AffineExpansionStorage(f)
        # Genereate random theta
        theta = RandomTuple(self.Q)
        # Return
        return (theta, F)

    def evaluate_builtin(self, theta, F):
        result_builtin = F[0].copy()
        result_builtin.zero()
        for i in range(self.Q):
            result_builtin.add_local(theta[i] * F[i].get_local())
        result_builtin.apply("insert")
        return result_builtin

    def evaluate_backend(self, theta, F):
        return sum(product(theta, F))

    def assert_backend(self, theta, F, result_backend):
        result_builtin = self.evaluate_builtin(theta, F)
        relative_error = (result_builtin - result_backend).norm("l2") / result_builtin.norm("l2")
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(3, 7)])
@pytest.mark.parametrize("N", [10 + 4 * j for j in range(1, 4)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_product.keys()))
def test_dolfin_vector_assembly(Th, N, test_type, benchmark):
    data = Data(Th, N)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()) + ", N = " + str(N))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global product, sum
        product, sum = all_product[test_type], all_sum[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
