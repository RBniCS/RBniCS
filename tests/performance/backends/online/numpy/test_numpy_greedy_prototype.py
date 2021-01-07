# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import builtins
import pytest
from numpy import einsum, isclose, zeros as legacy_tensor
from rbnics.backends import product as factory_product, sum as factory_sum, transpose as factory_transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, online_product, online_sum, online_transpose
from rbnics.backends.online.numpy import product as numpy_product, sum as numpy_sum, transpose as numpy_transpose
from test_numpy_utils import RandomNumber, RandomNumpyMatrix, RandomNumpyVector, RandomTuple

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
        self.Ntrain = 10

    def generate_random(self):
        aa_product = OnlineAffineExpansionStorage(self.Qa, self.Qa)
        aa_product_legacy = legacy_tensor((self.Qa, self.Qa, self.N, self.N))
        for i in range(self.Qa):
            for j in range(self.Qa):
                # Generate random matrix
                aa_product[i, j] = RandomNumpyMatrix(self.N, self.N)
                for n in range(self.N):
                    for m in range(self.N):
                        aa_product_legacy[i, j, n, m] = aa_product[i, j][n, m]
        af_product = OnlineAffineExpansionStorage(self.Qa, self.Qf)
        af_product_legacy = legacy_tensor((self.Qa, self.Qf, self.N))
        for i in range(self.Qa):
            for j in range(self.Qf):
                # Generate random matrix
                af_product[i, j] = RandomNumpyVector(self.N)
                for n in range(self.N):
                    af_product_legacy[i, j, n] = af_product[i, j][n]
        ff_product = OnlineAffineExpansionStorage(self.Qf, self.Qf)
        ff_product_legacy = legacy_tensor((self.Qf, self.Qf))
        for i in range(self.Qf):
            for j in range(self.Qf):
                # Generate random matrix
                ff_product[i, j] = RandomNumber()
                ff_product_legacy[i, j] = ff_product[i, j]
        # Genereate random theta
        theta_a = []
        theta_f = []
        for t in range(self.Ntrain):
            theta_a.append(RandomTuple(self.Qa))
            theta_f.append(RandomTuple(self.Qf))
        # Generate random solution
        u = []
        v = []
        for t in range(self.Ntrain):
            u.append(RandomNumpyVector(self.N))
            v.append(RandomNumpyVector(self.N))
        # Return
        return (theta_a, theta_f,
                aa_product, af_product, ff_product,
                aa_product_legacy, af_product_legacy, ff_product_legacy, u, v)

    def evaluate_builtin(self,
                         theta_a, theta_f,
                         aa_product, af_product, ff_product,
                         aa_product_legacy, af_product_legacy, ff_product_legacy,
                         u, v):
        result_builtin = list()
        for t in range(self.Ntrain):
            result_builtin.append(
                einsum("n,i,ijnm,j,m", u[t], theta_a[t], aa_product_legacy, theta_a[t], v[t], optimize=True)
                + einsum("i,ijn,j,n", theta_a[t], af_product_legacy, theta_f[t], u[t], optimize=True)
                + einsum("i,ij,j", theta_f[t], ff_product_legacy, theta_f[t], optimize=True)
            )
        return result_builtin

    def evaluate_backend(self,
                         theta_a, theta_f,
                         aa_product, af_product, ff_product,
                         aa_product_legacy, af_product_legacy, ff_product_legacy,
                         u, v):
        result_backend = []
        for t in range(self.Ntrain):
            result_backend.append(
                transpose(u[t]) * sum(product(theta_a[t], aa_product[:self.N, :self.N], theta_a[t])) * v[t]
                + transpose(u[t]) * sum(product(theta_a[t], af_product[:self.N], theta_f[t]))
                + sum(product(theta_f[t], ff_product, theta_f[t]))
            )
        return result_backend

    def assert_backend(self,
                       theta_a, theta_f,
                       aa_product, af_product, ff_product,
                       aa_product_legacy, af_product_legacy, ff_product_legacy,
                       u, v,
                       result_backend):
        assert len(result_backend) == self.Ntrain
        result_builtin = self.evaluate_builtin(
            theta_a, theta_f,
            aa_product, af_product, ff_product,
            aa_product_legacy, af_product_legacy, ff_product_legacy,
            u, v)
        assert len(result_builtin) == self.Ntrain
        relative_error = builtins.sum([
            abs(result_builtin_t - result_backend_t) / abs(result_builtin_t)
            for (result_builtin_t, result_backend_t) in zip(result_builtin, result_backend)]) / self.Ntrain
        assert isclose(relative_error, 0., atol=1e-10)


@pytest.mark.parametrize("N", [2**(i + 3) for i in range(1, 3)])
@pytest.mark.parametrize("Qa", [2 + 4 * j for j in range(1, 3)])
@pytest.mark.parametrize("Qf", [2 + 4 * k for k in range(1, 3)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_numpy_greedy_prototype(N, Qa, Qf, test_type, benchmark):
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
