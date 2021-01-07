# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import einsum, isclose, zeros as legacy_tensor
from rbnics.backends import product as factory_product, sum as factory_sum, transpose as factory_transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, online_product, online_sum, online_transpose
from rbnics.backends.online.numpy import product as numpy_product, sum as numpy_sum, transpose as numpy_transpose
from test_numpy_utils import RandomNumpyVector, RandomTuple

product = None
sum = None
transpose = None
all_product = {"numpy": numpy_product, "online": online_product, "factory": factory_product}
all_sum = {"numpy": numpy_sum, "online": online_sum, "factory": factory_sum}
all_transpose = {"numpy": numpy_transpose, "online": online_transpose, "factory": factory_transpose}


class Data(object):
    def __init__(self, N, Qa, Qf):
        self.N = N
        self.Qa = Qa
        self.Qf = Qf

    def generate_random(self):
        af_product = OnlineAffineExpansionStorage(self.Qa, self.Qf)
        af_product_legacy = legacy_tensor((self.Qa, self.Qf, self.N))
        for i in range(self.Qa):
            for j in range(self.Qf):
                # Generate random vector
                af_product[i, j] = RandomNumpyVector(self.N)
                for n in range(self.N):
                    af_product_legacy[i, j, n] = af_product[i, j][n]
        # Genereate random theta
        theta_a = RandomTuple(self.Qa)
        theta_f = RandomTuple(self.Qf)
        # Generate random solution
        u = RandomNumpyVector(self.N)
        # Return
        return (theta_a, theta_f, af_product, af_product_legacy, u)

    def evaluate_builtin(self, theta_a, theta_f, af_product, af_product_legacy, u):
        return einsum("i,ijn,j,n", theta_a, af_product_legacy, theta_f, u, optimize=True)

    def evaluate_backend(self, theta_a, theta_f, af_product, af_product_legacy, u):
        return transpose(u) * sum(product(theta_a, af_product, theta_f))

    def assert_backend(self, theta_a, theta_f, af_product, af_product_legacy, u, result_backend):
        result_builtin = self.evaluate_builtin(theta_a, theta_f, af_product, af_product_legacy, u)
        relative_error = abs(result_builtin - result_backend) / abs(result_builtin)
        assert isclose(relative_error, 0., atol=1e-10)


@pytest.mark.parametrize("N", [2**(i + 3) for i in range(1, 3)])
@pytest.mark.parametrize("Qa", [2 + 4 * j for j in range(1, 3)])
@pytest.mark.parametrize("Qf", [2 + 4 * k for k in range(1, 3)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_numpy_error_estimation_af_evaluation(N, Qa, Qf, test_type, benchmark):
    data = Data(N, Qa, Qf)
    print("N = " + str(N) + ", Qa = " + str(Qa) + ", Qf = " + str(Qf))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global product, sum, transpose
        product, sum, transpose = all_product[test_type], all_sum[test_type], all_transpose[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
