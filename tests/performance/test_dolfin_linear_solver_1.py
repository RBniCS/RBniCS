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
from rbnics.backends import LinearSolver as FactoryLinearSolver
from rbnics.backends.dolfin import LinearSolver as DolfinLinearSolver
LinearSolver = None
AllLinearSolver = {"dolfin": FactoryLinearSolver, "factory": DolfinLinearSolver}
from rbnics.utils.io import Timer

for i in range(3, 10):
    Nh = 2**i
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nh, Nh)
    V = FunctionSpace(mesh, "Lagrange", 1)
    print("Nh =", V.dim())
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    g = Expression("sin(x[0])*cos(x[1])", element=V.ufl_element())
    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    f = g*v*dx
    
    # Prepare timer
    timer = Timer("parallel")
        
    # Test using builtin solve
    solution_1a = Function(V)
    timer.start()
    solve(a == f, solution_1a, solver_parameters={"linear_solver": "mumps"})
    sec_1a = timer.stop()
    print("Builtin method:", sec_1a, "sec")
    
    for backend in ("dolfin", "factory"):
        print("Testing", backend, "backend")
        LinearSolver = AllLinearSolver[backend]
        
        solution_1b = Function(V)
        timer.start()
        solver = LinearSolver(a, solution_1b, f)
        solver.solve()
        sec_1b = timer.stop()
        print("\tLinearSolver class:", sec_1b)
        
        print("\tRelative overhead of the LinearSolver class:", (sec_1b - sec_1a)/sec_1a)
        
        error = Function(V)
        error.vector().add_local(+ solution_1a.vector().array())
        error.vector().add_local(- solution_1b.vector().array())
        error.vector().apply("add")
        print("\tRelative error:", error.vector().norm("l2"))
