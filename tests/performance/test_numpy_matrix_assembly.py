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

from __future__ import print_function
from numpy.linalg import norm
from rbnics.backends import product, sum
from rbnics.backends.online import OnlineAffineExpansionStorage
from test_utils import RandomNumpyMatrix, RandomTuple, TestBase

class Test(TestBase):
    def __init__(self, N, Q):
        self.N = N
        self.Q = Q
        # Call parent init
        TestBase.__init__(self)
            
    def run(self):
        N = self.N
        Q = self.Q
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                A = OnlineAffineExpansionStorage(self.Q)
                for i in range(self.Q):
                    # Generate random matrix
                    A[i] = RandomNumpyMatrix(N, N)
                # Genereate random theta
                theta = RandomTuple(Q)
                # Store
                self.storage[self.index] = (theta, A)
            else:
                (theta, A) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                assembled_matrix_builtin = theta[0]*A[0]
                for i in range(1, self.Q):
                    assembled_matrix_builtin += theta[i]*A[i]
                assembled_matrix_builtin.M = N
                assembled_matrix_builtin.N = N
                assembled_matrix_builtin.backend = A[0].backend
                assembled_matrix_builtin.wrapping = A[0].wrapping
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using sum(product()) method
                assembled_matrix_sum_product = sum(product(theta, A))
        if test_id >= 2:
            return norm(assembled_matrix_builtin - assembled_matrix_sum_product)/norm(assembled_matrix_builtin)

for i in range(4, 9):
    N = 2**i
    for j in range(1, 4):
        Q = 10 + 4*j
        test = Test(N, Q)
        print("N =", N, "and Q =", Q)
        
        test.init_test(0)
        (usec_0_build, usec_0_access) = test.timeit()
        print("Construction:", usec_0_build, "usec", "(number of runs: ", test.number_of_runs(), ")")
        print("Access:", usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "a")
        usec_1a = test.timeit()
        print("Builtin method:", usec_1a - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "b")
        usec_1b = test.timeit()
        print("sum(product()) method:", usec_1b - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        print("Relative overhead of the sum(product()) method:", (usec_1b - usec_1a)/(usec_1a - usec_0_access))
        
        test.init_test(2)
        error = test.average()
        print("Relative error:", error)
    
