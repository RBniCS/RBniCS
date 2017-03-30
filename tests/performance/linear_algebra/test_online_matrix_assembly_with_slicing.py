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
## @file test_truth_matrix_assembly.py
#  @brief Test sum_{i = 1}^{Q} theta_i A_i
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from test_main import TestBase
from dolfin import *
from rbnics.backends import product, sum
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineMatrix
from numpy.linalg import norm
from numpy.random import randint

OnlineMatrix_Type = OnlineMatrix.Type()

class Test(TestBase):
    def __init__(self, Nmax, Q):
        self.Nmax = Nmax
        self.Q = Q
        # Call parent init
        TestBase.__init__(self)
            
    def run(self):
        Nmax = self.Nmax
        Q = self.Q
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                A = OnlineAffineExpansionStorage(self.Q)
                for i in range(self.Q):
                    # Generate random matrix
                    A[i] = OnlineMatrix_Type(self.rand(Nmax, Nmax))
                # Genereate random theta
                theta = tuple(self.rand(Q))
                # Generate N <= Nmax
                N = randint(1, Nmax + 1)
                # Store
                self.storage[self.index] = (theta, A, N)
            else:
                (theta, A, N) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                assembled_matrix_builtin = theta[0]*A[0][:N, :N]
                for i in range(1, self.Q):
                    assembled_matrix_builtin += theta[i]*A[i][:N, :N]
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using sum(product()) method
                assembled_matrix_sum_product = sum(product(theta, A[:N, :N]))
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
    
