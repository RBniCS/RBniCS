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
from RBniCS.backends import product, sum, transpose
from RBniCS.backends.online import OnlineAffineExpansionStorage, OnlineMatrix, OnlineVector
from numpy import zeros as legacy_tensor
from numpy.linalg import norm

OnlineMatrix_Type = OnlineMatrix.Type()
OnlineVector_Type = OnlineVector.Type()

class Test(TestBase):
    def __init__(self, N, Qa, Qf):
        self.N = N
        self.Qa = Qa
        self.Qf = Qf
        # Call parent init
        TestBase.__init__(self)
            
    def run(self):
        N = self.N
        Qa = self.Qa
        Qf = self.Qf
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                af_product = OnlineAffineExpansionStorage(Qa, Qf)
                af_product_legacy = legacy_tensor((Qa, Qf, N))
                for i in range(Qa):
                    for j in range(Qf):
                        # Generate random matrix
                        af_product[i, j] = OnlineVector_Type(self.rand(N)).transpose()
                        for n in range(N):
                            af_product_legacy[i, j, n] = af_product[i, j][n]
                # Genereate random theta
                theta_a = tuple(self.rand(Qa))
                theta_f = tuple(self.rand(Qf))
                # Generate random solution
                u = OnlineVector_Type(self.rand(N)).transpose() # as column vector
                # Store
                self.storage[self.index] = (theta_a, theta_f, af_product, af_product_legacy, u)
            else:
                (theta_a, theta_f, af_product, af_product_legacy, u) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                error_estimator_legacy = 0.
                for i in range(Qa):
                    for j in range(Qf):
                        for n in range(N):
                            error_estimator_legacy += theta_a[i]*af_product_legacy[i, j, n]*theta_f[j]*u.item(n)
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using sum(product()) method
                error_estimator_sum_product = transpose(u)*sum(product(theta_a, af_product, theta_f))
        if test_id >= 2:
            return abs(error_estimator_legacy - error_estimator_sum_product)/abs(error_estimator_legacy)

for i in range(4, 9):
    N = 2**i
    for j in range(1, 8):
        Qa = 2 + 4*j
        for k in range(1, 8):
            Qf = 2 + 4*k
            test = Test(N, Qa, Qf)
            print("N =", N, "and Qa =", Qa, "and Qf =", Qf)
            
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
    
