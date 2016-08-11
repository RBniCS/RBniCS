# Copyright (C) 2015-2016 by the RBniCS authors
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
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type
from RBniCS.linear_algebra.sum import sum
from RBniCS.linear_algebra.product import product
from RBniCS.linear_algebra.affine_expansion_online_storage import AffineExpansionOnlineStorage
from numpy import zeros as legacy_tensor
from numpy.linalg import norm

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
                ff_product = AffineExpansionOnlineStorage(Q, Q)
                ff_product_legacy = legacy_tensor((Q, Q))
                for i in range(Q):
                    for j in range(Q):
                        # Generate random matrix
                        ff_product[i, j] = self.rand(1)[0]
                        ff_product_legacy[i, j] = ff_product[i, j]
                # Genereate random theta
                theta = tuple(self.rand(Q))
                # Store
                self.storage[self.index] = (theta, ff_product, ff_product_legacy)
            else:
                (theta, ff_product, ff_product_legacy) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                error_estimator_legacy = 0.
                for i in range(Q):
                    for j in range(Q):
                        error_estimator_legacy += theta[i]*ff_product_legacy[i, j]*theta[j]
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using sum(product()) method
                error_estimator_sum_product = sum(product(theta, ff_product, theta))
        if test_id >= 2:
            return abs(error_estimator_legacy - error_estimator_sum_product)/abs(error_estimator_legacy)

for i in range(4, 9):
    N = 2**i
    for j in range(1, 8):
        Q = 2 + 4*j
        test = Test(N, Q)
        print("N =", N, "and Q =", Q)
        
        test.init_test(0)
        (usec_0_build, usec_0_access) = test.timeit()
        print("Construction:", usec_0_build, "usec", "(number of runs: ", test.number_of_runs(), ")")
        print("Access:", usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "a")
        usec_1a = test.timeit()
        print("Legacy method:", usec_1a - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "b")
        usec_1b = test.timeit()
        print("sum(product()) method:", usec_1b - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        print("Speed up of the sum(product()) method:", (usec_1a - usec_0_access)/(usec_1b - usec_0_access))
        
        test.init_test(2)
        error = test.average()
        print("Relative error:", error)
    
