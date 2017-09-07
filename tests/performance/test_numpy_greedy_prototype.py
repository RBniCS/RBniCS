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


from numpy import zeros as legacy_tensor
from numpy.linalg import norm
from rbnics.backends import product, sum, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage
from test_utils import RandomNumber, RandomNumpyMatrix, RandomNumpyVector, RandomTuple, TestBase

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
        Ntrain = 100
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                aa_product = OnlineAffineExpansionStorage(Qa, Qa)
                aa_product_legacy = legacy_tensor((Qa, Qa, N, N))
                for i in range(Qa):
                    for j in range(Qa):
                        # Generate random matrix
                        aa_product[i, j] = RandomNumpyMatrix(N, N)
                        for n in range(N):
                            for m in range(N):
                                aa_product_legacy[i, j, n, m] = aa_product[i, j][n, m]
                af_product = OnlineAffineExpansionStorage(Qa, Qf)
                af_product_legacy = legacy_tensor((Qa, Qf, N))
                for i in range(Qa):
                    for j in range(Qf):
                        # Generate random matrix
                        af_product[i, j] = RandomNumpyVector(N)
                        for n in range(N):
                            af_product_legacy[i, j, n] = af_product[i, j][n]
                ff_product = OnlineAffineExpansionStorage(Qf, Qf)
                ff_product_legacy = legacy_tensor((Qf, Qf))
                for i in range(Qf):
                    for j in range(Qf):
                        # Generate random matrix
                        ff_product[i, j] = RandomNumber()
                        ff_product_legacy[i, j] = ff_product[i, j]
                # Genereate random theta
                theta_a = []
                theta_f = []
                for t in range(Ntrain):
                    theta_a.append(RandomTuple(Qa))
                    theta_f.append(RandomTuple(Qf))
                # Generate random solution
                u = []
                v = []
                for t in range(Ntrain):
                    u.append(RandomNumpyVector(N))
                    v.append(RandomNumpyVector(N))
                # Store
                self.storage[self.index] = (theta_a, theta_f, aa_product, af_product, ff_product, aa_product_legacy, af_product_legacy, ff_product_legacy, u, v)
            else:
                (theta_a, theta_f, aa_product, af_product, ff_product, aa_product_legacy, af_product_legacy, ff_product_legacy, u, v) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                all_error_estimator_legacy = []
                for t in range(Ntrain):
                    error_estimator_legacy = 0.
                    for i in range(Qa):
                        for j in range(Qa):
                            for n in range(N):
                                for m in range(N):
                                    error_estimator_legacy += u[t].item(n)*theta_a[t][i]*aa_product_legacy[i, j, n, m]*theta_a[t][j]*v[t].item(m)
                    for i in range(Qa):
                        for j in range(Qf):
                            for n in range(N):
                                error_estimator_legacy += theta_a[t][i]*af_product_legacy[i, j, n]*theta_f[t][j]*u[t].item(n)
                    for i in range(Qf):
                        for j in range(Qf):
                            error_estimator_legacy += theta_f[t][i]*ff_product_legacy[i, j]*theta_f[t][j]
                    all_error_estimator_legacy.append(error_estimator_legacy)
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using sum(product()) method
                all_error_estimator_sum_product = []
                for t in range(Ntrain):
                    all_error_estimator_sum_product.append(
                        transpose(u[t])*sum(product(theta_a[t], aa_product[:N, :N], theta_a[t]))*v[t] +
                        transpose(u[t])*sum(product(theta_a[t], af_product[:N], theta_f[t])) +
                        sum(product(theta_f[t], ff_product, theta_f[t]))
                    )
        if test_id >= 2:
            average_relative_error = 0.
            for t in range(Ntrain):
                average_relative_error += abs(all_error_estimator_legacy[t] - all_error_estimator_sum_product[t])/abs(all_error_estimator_legacy[t])
            average_relative_error /= Ntrain
            return average_relative_error

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
    
