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
from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.transpose import transpose

class Test(TestBase):
    def __init__(self, N, test_id, test_subid=None):
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, "Lagrange", 1)
        self.v1 = Function(V)
        self.v2 = Function(V)
        self.k = Function(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        self.a = self.k*inner(grad(u), grad(v))*dx
        # Call parent init
        TestBase.__init__(self, N, test_id, test_subid)
            
    def run(self):
        N = self.N
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            # Generate random vectors
            self.v1.vector()[:] = self.rand(self.v1.vector().size())
            self.v2.vector()[:] = self.rand(self.v2.vector().size())
            self.k.vector()[:] = self.rand(self.k.vector().size())
            # Generate random matrix
            A = assemble(self.a)
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                v1_dot_A_v2_builtin = self.v1.vector().inner(A*self.v2.vector())
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using transpose() method
                v1_dot_A_v2_transpose = transpose(self.v1.vector())*A*self.v2.vector()
        if test_id >= 2:
            return (v1_dot_A_v2_builtin - v1_dot_A_v2_transpose)/v1_dot_A_v2_builtin

for i in range(1, 8):
    N = 2**i
    
    test = Test(N, 0)
    print("N =", test.v1.vector().size())
    usec_0 = test.timeit()
    print("Construction:", usec_0, "usec", "(number of runs: ", test.number_of_runs(), ")")
    
    test = Test(N, 1, "a")
    usec_1a = test.timeit()
    print("Builtin method:", usec_1a - usec_0, "usec", "(number of runs: ", test.number_of_runs(), ")")
    
    test = Test(N, 1, "b")
    usec_1b = test.timeit()
    print("transpose() method:", usec_1b - usec_0, "usec", "(number of runs: ", test.number_of_runs(), ")")
    
    print("Relative overhead of the transpose() method:", (usec_1b - usec_1a)/(usec_1a - usec_0))
    
    test = Test(N, 2)
    error = test.average()
    print("Relative error:", error)
    
