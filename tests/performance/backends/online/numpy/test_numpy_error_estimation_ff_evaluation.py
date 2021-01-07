# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import einsum, isclose, zeros as legacy_tensor
from rbnics.backends import product as factory_product, sum as factory_sum
from rbnics.backends.online import OnlineAffineExpansionStorage, online_product, online_sum
from rbnics.backends.online.numpy import product as numpy_product, sum as numpy_sum
from test_numpy_utils import RandomNumber, RandomTuple

product = None
sum = None
all_product = {"numpy": numpy_product, "online": online_product, "factory": factory_product}
all_sum = {"numpy": numpy_sum, "online": online_sum, "factory": factory_sum}


class Data(object):
    def __init__(self, N, Q):
        self.N = N
        self.Q = Q

    def generate_random(self):
        ff_product = OnlineAffineExpansionStorage(self.Q, self.Q)
        ff_product_legacy = legacy_tensor((self.Q, self.Q))
        for i in range(self.Q):
            for j in range(self.Q):
                # Generate random matrix
                ff_product[i, j] = RandomNumber()
                ff_product_legacy[i, j] = ff_product[i, j]
        # Genereate random theta
        theta = RandomTuple(self.Q)
        # Return
        return (theta, ff_product, ff_product_legacy)

    def evaluate_builtin(self, theta, ff_product, ff_product_legacy):
        return einsum("i,ij,j", theta, ff_product_legacy, theta, optimize=True)

    def evaluate_backend(self, theta, ff_product, ff_product_legacy):
        return sum(product(theta, ff_product, theta))

    def assert_backend(self, theta, ff_product, ff_product_legacy, result_backend):
        result_builtin = self.evaluate_builtin(theta, ff_product, ff_product_legacy)
        relative_error = abs(result_builtin - result_backend) / abs(result_builtin)
        assert isclose(relative_error, 0., atol=1e-10)


@pytest.mark.parametrize("N", [2**(i + 3) for i in range(1, 3)])
@pytest.mark.parametrize("Q", [2 + 4 * j for j in range(1, 3)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_product.keys()))
def test_numpy_error_estimation_ff_evaluation(N, Q, test_type, benchmark):
    data = Data(N, Q)
    print("N = " + str(N) + ", Q = " + str(Q))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global product, sum
        product, sum = all_product[test_type], all_sum[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
