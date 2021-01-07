# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import dot, isclose
from rbnics.backends import transpose as factory_transpose
from rbnics.backends.online import online_transpose
from rbnics.backends.online.numpy import transpose as numpy_transpose
from test_numpy_utils import RandomNumpyMatrix, RandomNumpyVector

transpose = None
all_transpose = {"numpy": numpy_transpose, "online": online_transpose, "factory": factory_transpose}


class Data(object):
    def __init__(self, N):
        self.N = N

    def generate_random(self):
        # Generate random vectors
        v1 = RandomNumpyVector(self.N)
        v2 = RandomNumpyVector(self.N)
        # Generate random matrix
        A = RandomNumpyMatrix(self.N, self.N)
        # Return
        return (v1, v2, A)

    def evaluate_builtin(self, v1, v2, A):
        return float(dot(v1, dot(A, v2)))

    def evaluate_backend(self, v1, v2, A):
        return transpose(v1) * A * v2

    def assert_backend(self, v1, v2, A, result_backend):
        result_builtin = self.evaluate_builtin(v1, v2, A)
        relative_error = (result_builtin - result_backend) / result_builtin
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("N", [2**i for i in range(1, 9)])
@pytest.mark.parametrize("test_type", ["builtin"] + list(all_transpose.keys()))
def test_numpy_v1_dot_A_v2(N, test_type, benchmark):
    data = Data(N)
    print("N = " + str(N))
    if test_type == "builtin":
        print("Testing", test_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing", test_type, "backend")
        global transpose
        transpose = all_transpose[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
