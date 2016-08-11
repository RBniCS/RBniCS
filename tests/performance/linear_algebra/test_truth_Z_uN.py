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
## @file test_v1_dot_v2.py
#  @brief Test v1 dot v2 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from test_main import TestBase
from dolfin import *
from RBniCS.linear_algebra.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.linear_algebra.online_vector import OnlineVector_Type
from RBniCS.linear_algebra.transpose import transpose

class Test(TestBase):
    def __init__(self, Nh, N):
        self.N = N
        mesh = UnitSquareMesh(Nh, Nh)
        V = FunctionSpace(mesh, "Lagrange", 1)
        self.b = Function(V)
        self.Z = BasisFunctionsMatrix()
        # Call parent init
        TestBase.__init__(self)
            
    def run(self):
        N = self.N
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                # Generate random vectors
                self.Z = BasisFunctionsMatrix()
                for _ in range(self.N):
                    self.b.vector().set_local(self.rand(self.b.vector().array().size))
                    self.b.vector().apply("insert")
                    self.Z.enrich(self.b)
                uN = OnlineVector_Type(self.rand(N)).transpose() # as column vector
                # Store
                self.storage[self.index] = (self.Z, uN)
            else:
                (self.Z, uN) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                Z_uN_builtin = uN.item(0)*self.Z[0]
                for i in range(1, self.N):
                    Z_uN_builtin.add_local(uN.item(i)*self.Z[i].array())
                Z_uN_builtin.apply("add")
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using mul method
                Z_uN_mul = self.Z*uN
        if test_id >= 2:
            return (Z_uN_builtin - Z_uN_mul).norm("l2")/Z_uN_builtin.norm("l2")

for i in range(3, 7):
    Nh = 2**i
    for j in range(1, 4):
        N = 10 + 4*j
        test = Test(Nh, N)
        print("Nh =", test.b.vector().size(), "and N =", N)
        
        test.init_test(0)
        (usec_0_build, usec_0_access) = test.timeit()
        print("Construction:", usec_0_build, "usec", "(number of runs: ", test.number_of_runs(), ")")
        print("Access:", usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "a")
        usec_1a = test.timeit()
        print("Builtin method:", usec_1a - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "b")
        usec_1b = test.timeit()
        print("mul method:", usec_1b - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        print("Relative overhead of the transpose() method:", (usec_1b - usec_1a)/(usec_1a - usec_0_access))
        
        test.init_test(2)
        error = test.average()
        print("Relative error:", error)
    
