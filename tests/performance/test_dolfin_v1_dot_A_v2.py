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


from dolfin import *
from rbnics.backends import transpose as factory_transpose
from rbnics.backends.dolfin import transpose as dolfin_transpose
transpose = None
all_transpose = {"dolfin": dolfin_transpose, "factory": factory_transpose}
from test_utils import RandomDolfinFunction, TestBase

class Test(TestBase):
    def __init__(self, Nh):
        mesh = UnitSquareMesh(Nh, Nh)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.a = lambda k: k*inner(grad(u), grad(v))*dx
        # Call parent init
        TestBase.__init__(self)
            
    def run(self):
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                # Generate random vectors
                v1 = RandomDolfinFunction(self.V).vector()
                v2 = RandomDolfinFunction(self.V).vector()
                k = RandomDolfinFunction(self.V)
                # Generate random matrix
                A = assemble(self.a(k))
                # Store
                self.storage[self.index] = (v1, v2, A)
            else:
                (v1, v2, A) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                v1_dot_A_v2_builtin = v1.inner(A*v2)
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using transpose() method
                v1_dot_A_v2_transpose = transpose(v1)*A*v2
        if test_id >= 2:
            return (v1_dot_A_v2_builtin - v1_dot_A_v2_transpose)/v1_dot_A_v2_builtin

for i in range(1, 9):
    Nh = 2**i
    test = Test(Nh)
    print("Nh =", test.V.dim())
    
    test.init_test(0)
    (usec_0_build, usec_0_access) = test.timeit()
    print("Construction:", usec_0_build, "usec", "(number of runs: ", test.number_of_runs(), ")")
    print("Access:", usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
    
    test.init_test(1, "a")
    usec_1a = test.timeit()
    print("Builtin method:", usec_1a - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
    
    for backend in ("dolfin", "factory"):
        print("Testing", backend, "backend")
        transpose = all_transpose[backend]
        
        test.init_test(1, "b")
        usec_1b = test.timeit()
        print("\ttranspose() method:", usec_1b - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        print("\tRelative overhead of the transpose() method:", (usec_1b - usec_1a)/(usec_1a - usec_0_access))
        
        test.init_test(2)
        error = test.average()
        print("\tRelative error:", error)
