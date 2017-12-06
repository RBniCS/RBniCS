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
from numpy import einsum, isclose, zeros as legacy_tensor
from rbnics.backends import product as factory_product, sum as factory_sum, transpose as factory_transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, online_product, online_sum, online_transpose
from rbnics.backends.online.numpy import product as numpy_product, sum as numpy_sum, transpose as numpy_transpose
from test_utils import RandomNumpyMatrix, RandomNumpyVector, RandomTuple

product = None
sum = None
transpose = None
all_product = {"numpy": numpy_product, "online": online_product, "factory": factory_product}
all_sum = {"numpy": numpy_sum, "online": online_sum, "factory": factory_sum}
all_transpose = {"numpy": numpy_transpose, "online": online_transpose, "factory": factory_transpose}

class Data(object):
    def __init__(self, N, Q):
        self.N = N
        self.Q = Q
        
    def generate_random(self):
        aa_product = OnlineAffineExpansionStorage(self.Q, self.Q)
        aa_product_legacy = legacy_tensor((self.Q, self.Q, self.N, self.N))
        for i in range(self.Q):
            for j in range(self.Q):
                # Generate random matrix
                aa_product[i, j] = RandomNumpyMatrix(self.N, self.N)
                for n in range(self.N):
                    for m in range(self.N):
                        aa_product_legacy[i, j, n, m] = aa_product[i, j][n, m]
        # Genereate random theta
        theta = RandomTuple(self.Q)
        # Generate random solution
        u = RandomNumpyVector(self.N)
        v = RandomNumpyVector(self.N)
        # Return
        return (theta, aa_product, aa_product_legacy, u, v)
        
    def evaluate_builtin(self, theta, aa_product, aa_product_legacy, u, v):
        return einsum("n,i,ijnm,j,m", u, theta, aa_product_legacy, theta, v, optimize=True)
        
    def evaluate_backend(self, theta, aa_product, aa_product_legacy, u, v):
        return transpose(u)*sum(product(theta, aa_product, theta))*v
        
    def assert_backend(self, theta, aa_product, aa_product_legacy, u, v, result_backend):
        result_builtin = self.evaluate_builtin(theta, aa_product, aa_product_legacy, u, v)
        relative_error = abs(result_builtin - result_backend)/abs(result_builtin)
        assert isclose(relative_error, 0., atol=1e-10)

@pytest.mark.parametrize("N", [2**(i + 3) for i in range(1, 3)])
@pytest.mark.parametrize("Q", [2 + 4*j for j in range(1, 3)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_numpy_error_estimation_aa_evaluation(N, Q, test_type, benchmark):
    data = Data(N, Q)
    print("N = " + str(N) + ", Q = " + str(Q))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global product, sum, transpose
        product, sum, transpose = all_product[test_type], all_sum[test_type], all_transpose[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
