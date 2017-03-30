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
from rbnics.backends.fenics import TimeStepping as SparseTimeStepping
#from rbnics.backends.numpy import Function as DenseFunction, Matrix as DenseMatrix, TimeStepping as DenseTimeStepping, Vector as DenseVector

"""
Solve
    u_tt - ((1 + u^2) u_x)_x = g,   (t, x) in [0, 1] x [0, 2*pi]
    u = sin(t),                     (t, x) in [0, 1] x {0, 2*pi}
    u = sin(x),                     (t, x) in {0}    x [0, 2*pi]
    u_t = cos(x),                   (t, x) in {0}    x [0, 2*pi]
for g such that u = u_ex = sin(x+t)
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

# Define exact solution dot dot
exact_solution_dot_dot_expression = Expression("-sin(x[0]+t)", t=0, element=V.ufl_element())
# ... and interpolate it at the final time
exact_solution_dot_dot_expression.t = T
exact_solution_dot_dot = project(exact_solution_dot_dot_expression, V)

# Define variational problem
du = TrialFunction(V)
du_dot_dot = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
u_dot_dot = Function(V)
g = Expression("-1./2.*sin(t+x[0])*(3*cos(2*(t+x[0]))+1)", t=0., element=V.ufl_element())
r_u = inner((1+u**2)*grad(u), grad(v))*dx
j_u = derivative(r_u, u, du)
r_u_dot_dot = inner(u_dot_dot, v)*dx
j_u_dot_dot = derivative(r_u_dot_dot, u_dot_dot, du_dot_dot)
r = r_u_dot_dot + r_u  - g*v*dx
x = inner(du, v)*dx

# Assemble inner product matrix
X = assemble(x)

# ~~~ Sparse case ~~~ #
# Residual and jacobian functions
def sparse_residual_eval(t, solution, solution_dot, solution_dot_dot):
    g.t = t
    return replace(r, {u: solution, u_dot_dot: solution_dot_dot})
def sparse_jacobian_eval(t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient):
    return (
        Constant(solution_dot_dot_coefficient)*replace(j_u_dot_dot, {u_dot_dot: solution_dot_dot}) +
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
exact_solution_dot_expression.t = 0.
sparse_solution = project(exact_solution_expression, V)
sparse_solution_dot = project(exact_solution_dot_expression, V)
sparse_solver = SparseTimeStepping(sparse_jacobian_eval, sparse_solution, sparse_residual_eval, sparse_bc_eval, time_order=2, solution_dot=sparse_solution_dot)
sparse_solver.set_parameters({
    "initial_time": 0.0,
    "time_step_size": dt,
    "final_time": T,
    "exact_final_time": "stepover",
    "integrator_type": "alpha2",
    "problem_type": "nonlinear",
    "snes_solver": {
        "linear_solver": "mumps",
        "maximum_iterations": 20,
        "report": True
    },
    "monitor": sparse_monitor,
    "report": True
})
all_sparse_solutions_time, all_sparse_solutions, all_sparse_solutions_dot, all_sparse_solutions_dot_dot = sparse_solver.solve()
assert len(all_sparse_solutions_time) == int(T/dt + 1)
assert len(all_sparse_solutions) == int(T/dt + 1)
assert len(all_sparse_solutions_dot) == int(T/dt + 1)
assert len(all_sparse_solutions_dot_dot) == int(T/dt + 1)

# Compute the error
sparse_error = Function(V)
sparse_error.vector().add_local(+ sparse_solution.vector().array())
sparse_error.vector().add_local(- exact_solution.vector().array())
sparse_error.vector().apply("")
sparse_error_norm = sparse_error.vector().inner(X*sparse_error.vector())
sparse_error_dot = Function(V)
sparse_error_dot.vector().add_local(+ all_sparse_solutions_dot[-1].vector().array())
sparse_error_dot.vector().add_local(- exact_solution_dot.vector().array())
sparse_error_dot.vector().apply("")
sparse_error_dot_norm = sparse_error_dot.vector().inner(X*sparse_error_dot.vector())
sparse_error_dot_dot = Function(V)
sparse_error_dot_dot.vector().add_local(+ all_sparse_solutions_dot_dot[-1].vector().array())
sparse_error_dot_dot.vector().add_local(- exact_solution_dot_dot.vector().array())
sparse_error_dot_dot.vector().apply("")
sparse_error_dot_dot_norm = sparse_error_dot_dot.vector().inner(X*sparse_error_dot_dot.vector())
print "SparseTimeStepping error:", sparse_error_norm, sparse_error_dot_norm, sparse_error_dot_dot_norm
assert isclose(sparse_error_norm, 0., atol=1.e-5)
assert isclose(sparse_error_dot_norm, 0., atol=1.e-5)
assert isclose(sparse_error_dot_dot_norm, 0., atol=1.e-4)

