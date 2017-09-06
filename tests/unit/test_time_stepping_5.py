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

import sys
from numpy import asarray, isclose
from dolfin import *
import matplotlib.pyplot as plt
from rbnics.backends.abstract import TimeDependentProblem1Wrapper
from rbnics.backends.dolfin import TimeStepping as SparseTimeStepping

# Additional command line options for PETSc TS
PETScOptions().set("ts_bdf_order", "3")
PETScOptions().set("ts_bdf_adapt", "true")

"""
Solve
    u_t - u_xx = g,   (t, x) in [0, 1] x [0, 2*pi]
    u = sin(t),       (t, x) in [0, 1] x {0, 2*pi}
    u = sin(x),       (t, x) in {0}    x [0, 2*pi]
for g such that u = u_ex = sin(x+t)

The difference between this test and test_time_stepping_1 is that residual and jacobian eval functions return assembled tensors,
rather than forms.
"""

# Create mesh and define function space
mesh = IntervalMesh(132, 0, 2*pi)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 2*pi)
def boundary(x):
    return x[0] < 0 + DOLFIN_EPS or x[0] > 2*pi - 10*DOLFIN_EPS
    
# Define time step
dt = 0.01
T = 1.

# Define exact solution
exact_solution_expression = Expression("sin(x[0]+t)", t=0, element=V.ufl_element())
# ... and interpolate it at the final time
exact_solution_expression.t = T
exact_solution = project(exact_solution_expression, V)

# Define exact solution dot
exact_solution_dot_expression = Expression("cos(x[0]+t)", t=0, element=V.ufl_element())
# ... and interpolate it at the final time
exact_solution_dot_expression.t = T
exact_solution_dot = project(exact_solution_dot_expression, V)

# Define variational problem
du = TrialFunction(V)
du_dot = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
u_dot = Function(V)
g = Expression("sin(x[0]+t) + cos(x[0]+t)", t=0., element=V.ufl_element())
r_u = inner(grad(u), grad(v))*dx
j_u = derivative(r_u, u, du)
r_u_dot = inner(u_dot, v)*dx
j_u_dot = derivative(r_u_dot, u_dot, du_dot)
r = r_u_dot + r_u  - g*v*dx
x = inner(du, v)*dx
bc = [DirichletBC(V, exact_solution_expression, boundary)]

# Assemble inner product matrix
X = assemble(x)

# ~~~ Sparse case ~~~ #
class SparseProblemWrapper(TimeDependentProblem1Wrapper):
    # Residual and jacobian functions
    def residual_eval(self, t, solution, solution_dot):
        g.t = t
        return assemble(r)
    def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
        return assemble(j_u_dot)*solution_dot_coefficient + assemble(j_u)
        
    # Define boundary condition
    def bc_eval(self, t):
        exact_solution_expression.t = t
        return bc
        
    # Define initial condition
    def ic_eval(self):
        exact_solution_expression.t = 0.
        return project(exact_solution_expression, V)
        
    # Define custom monitor to plot the solution
    def monitor(self, t, solution, solution_dot):
        plt.subplot(1, 2, 1).clear()
        plot(solution, title="u at t = " + str(t))
        plt.subplot(1, 2, 2).clear()
        plot(solution_dot, title="u_dot at t = " + str(t))
        plt.show(block=False)
        plt.pause(DOLFIN_EPS)

for integrator_type in ("beuler", "bdf"):
    # Solve the time dependent problem
    sparse_problem_wrapper = SparseProblemWrapper()
    (sparse_solution, sparse_solution_dot) = (u, u_dot)
    sparse_solver = SparseTimeStepping(sparse_problem_wrapper, sparse_solution, sparse_solution_dot)
    sparse_solver.set_parameters({
        "initial_time": 0.0,
        "time_step_size": dt,
        "final_time": T,
        "exact_final_time": "stepover",
        "integrator_type": integrator_type,
        "problem_type": "linear",
        "linear_solver": "mumps",
        "monitor": sparse_problem_wrapper.monitor,
        "report": True
    })
    all_sparse_solutions_time, all_sparse_solutions, all_sparse_solutions_dot = sparse_solver.solve()
    assert len(all_sparse_solutions_time) == int(T/dt + 1)
    assert len(all_sparse_solutions) == int(T/dt + 1)
    assert len(all_sparse_solutions_dot) == int(T/dt + 1)

    # Compute the error
    sparse_error = Function(V)
    sparse_error.vector().add_local(+ sparse_solution.vector().array())
    sparse_error.vector().add_local(- exact_solution.vector().array())
    sparse_error.vector().apply("")
    sparse_error_norm = sparse_error.vector().inner(X*sparse_error.vector())
    sparse_error_dot = Function(V)
    sparse_error_dot.vector().add_local(+ sparse_solution_dot.vector().array())
    sparse_error_dot.vector().add_local(- exact_solution_dot.vector().array())
    sparse_error_dot.vector().apply("")
    sparse_error_dot_norm = sparse_error_dot.vector().inner(X*sparse_error_dot.vector())
    print "SparseTimeStepping error (" + integrator_type + "):", sparse_error_norm, sparse_error_dot_norm
    assert isclose(sparse_error_norm, 0., atol=1.e-4)
    assert isclose(sparse_error_dot_norm, 0., atol=1.e-4)

