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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import sys
from numpy import asarray, isclose
from dolfin import *
from RBniCS.backends.fenics import TimeStepping as SparseTimeStepping

# Additional command line options for PETSc TS
args = "--petsc.ts_bdf_order 3 --petsc.ts_bdf_adapt true"
parameters.parse(argv = sys.argv[0:1] + args.split())

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

# Assemble inner product matrix
X = assemble(x)

# ~~~ Sparse case ~~~ #
# Residual and jacobian functions
def sparse_residual_eval(t, solution, solution_dot):
    g.t = t
    return assemble(replace(r, {u: solution, u_dot: solution_dot}))
def sparse_jacobian_eval(t, solution, solution_dot, solution_dot_coefficient):
    return assemble(
        Constant(solution_dot_coefficient)*replace(j_u_dot, {u_dot: solution_dot}) +
        replace(j_u, {u: solution})
    )
    
# Define boundary condition
bc = [DirichletBC(V, exact_solution_expression, boundary)]
def sparse_bc_eval(t):
    exact_solution_expression.t = t
    return bc
    
# Define custom monitor to plot the solution
def sparse_monitor(t, solution):
    plot(solution, key="u", title="t = " + str(t))

# Solve the time dependent problem
exact_solution_expression.t = 0.
sparse_solution = project(exact_solution_expression, V)
sparse_solver = SparseTimeStepping(sparse_jacobian_eval, sparse_solution, sparse_residual_eval, sparse_bc_eval)
sparse_solver.set_parameters({
    "initial_time": 0.0,
    "time_step_size": dt,
    "final_time": T,
    "exact_final_time": "stepover",
    "integrator_type": "bdf",
    "problem_type": "linear",
    "linear_solver": "mumps",
    "monitor": sparse_monitor,
    "report": True
})
all_sparse_solutions = sparse_solver.solve()
assert len(all_sparse_solutions) == int(T/dt + 1)

# Compute the error
sparse_error = Function(V)
sparse_error.vector().add_local(+ sparse_solution.vector().array())
sparse_error.vector().add_local(- exact_solution.vector().array())
sparse_error.vector().apply("")
sparse_error_norm = sparse_error.vector().inner(X*sparse_error.vector())
print "SparseTimeStepping error:", sparse_error_norm
assert isclose(sparse_error_norm, 0., atol=1.e-5)

