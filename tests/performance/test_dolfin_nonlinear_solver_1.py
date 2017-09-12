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
from rbnics.backends import NonlinearSolver as FactoryNonlinearSolver
from rbnics.backends.abstract import NonlinearProblemWrapper
from rbnics.backends.dolfin import NonlinearSolver as DolfinNonlinearSolver
NonlinearSolver = None
AllNonlinearSolver = {"dolfin": FactoryNonlinearSolver, "factory": DolfinNonlinearSolver}
from rbnics.utils.io import Timer

for i in range(3, 10):
    Nh = 2**i
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nh, Nh)
    V = FunctionSpace(mesh, "Lagrange", 1)
    print("Nh =", V.dim())
    
    # Define variational problem
    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    g = Expression("sin(x[0])*cos(x[1])", element=V.ufl_element())
    r = inner(grad(u), grad(v))*dx + inner(u + u**3, v)*dx - g*v*dx
    j = derivative(r, u, du)
    
    class ProblemWrapper(NonlinearProblemWrapper):
        # Residual and jacobian functions
        def residual_eval(self, solution):
            return r
        def jacobian_eval(self, solution):
            return j
        
        # Define boundary condition
        def bc_eval(self):
            return None
    problem_wrapper = ProblemWrapper()
    
    # Define initial guess
    initial_guess_expression = Expression("0.1 + 0.9*x[0]*x[1]", element=V.ufl_element())
    
    # Prepare timer
    timer = Timer("parallel")
        
    # Test using builtin solve
    project(initial_guess_expression, V, function=u)
    timer.start()
    solve(r == 0, u, J=j,
        solver_parameters={
            "nonlinear_solver": "snes",
            "snes_solver": {
                "linear_solver": "mumps",
                "maximum_iterations": 20,
                "report": True,
                "error_on_nonconvergence": True
            }
        }
    )
    sec_1a = timer.stop()
    solution_1a = u.copy(deepcopy=True)
    print("Builtin method:", sec_1a, "sec")
    
    for backend in ("dolfin", "factory"):
        print("Testing", backend, "backend")
        NonlinearSolver = AllNonlinearSolver[backend]
        
        project(initial_guess_expression, V, function=u)
        timer.start()
        solver = NonlinearSolver(problem_wrapper, u)
        solver.set_parameters({
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "report": True,
            "error_on_nonconvergence": True
        })
        solver.solve()
        sec_1b = timer.stop()
        solution_1b = u.copy(deepcopy=True)
        print("\tNonlinearSolver class:", sec_1b)
        
        print("\tRelative overhead of the NonlinearSolver class:", (sec_1b - sec_1a)/sec_1a)
        
        error = Function(V)
        error.vector().add_local(+ solution_1a.vector().array())
        error.vector().add_local(- solution_1b.vector().array())
        error.vector().apply("add")
        print("\tRelative error:", error.vector().norm("l2"))
