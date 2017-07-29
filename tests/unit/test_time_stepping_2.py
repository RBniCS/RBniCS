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
from rbnics.backends.online.numpy import Function as DenseFunction, Matrix as DenseMatrix, TimeStepping as DenseTimeStepping, Vector as DenseVector

# Additional command line options for PETSc TS
args = "--petsc.ts_bdf_order 3 --petsc.ts_bdf_adapt true"
parameters.parse(argv = sys.argv[0:1] + args.split())

"""
Solve
    u_t - ((1 + u^2) u_x)_x = g,   (t, x) in [0, 1] x [0, 2*pi]
    u = sin(t),                    (t, x) in [0, 1] x {0, 2*pi}
    u = sin(x),                    (t, x) in {0}    x [0, 2*pi]
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

# Define variational problem
du = TrialFunction(V)
du_dot = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
u_dot = Function(V)
g = Expression("5./4.*sin(t+x[0])-3./4.*sin(3*(t+x[0]))+cos(t+x[0])", t=0., element=V.ufl_element())
r_u = inner((1+u**2)*grad(u), grad(v))*dx
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
        return r
    def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
        return Constant(solution_dot_coefficient)*j_u_dot + j_u
        
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
        "problem_type": "nonlinear",
        "snes_solver": {
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "report": True
        },
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

# ~~~ Dense case ~~~ #
if mesh.mpi_comm().size == 1: # dense solver is not partitioned
    x_to_dof = dict(zip(V.tabulate_dof_coordinates(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2*pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)
    
    class DenseProblemWrapper(TimeDependentProblem1Wrapper):
        # Residual and jacobian functions, reordering resulting matrix and vector
        # such that dof_0 and dof_2pi are in the first two rows/cols,
        # because the dense time stepping solver has implicitly this assumption
        def residual_eval(self, t, solution, solution_dot):
            self._solution_from_dense_to_sparse(solution, u)
            self._solution_from_dense_to_sparse(solution_dot, u_dot)
            g.t = t
            sparse_residual = assemble(r)
            dense_residual_array = sparse_residual.array()
            dense_residual_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_residual_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            dense_residual = DenseVector(*dense_residual_array.shape)
            dense_residual[:] = dense_residual_array.reshape((-1, 1))
            return dense_residual
            
        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            self._solution_from_dense_to_sparse(solution, u)
            self._solution_from_dense_to_sparse(solution_dot, u_dot)
            sparse_jacobian = assemble(Constant(solution_dot_coefficient)*j_u_dot + j_u)
            dense_jacobian_array = sparse_jacobian.array()
            dense_jacobian_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi], :] = dense_jacobian_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1], :]
            dense_jacobian_array[:, [0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_jacobian_array[:, [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            dense_jacobian = DenseMatrix(*dense_jacobian_array.shape)
            dense_jacobian[:] = dense_jacobian_array
            return dense_jacobian
        
        # Define boundary condition
        def bc_eval(self, t):
            return (sin(t), sin(t))
            
        # Define initial condition
        def ic_eval(self):
            exact_solution_expression.t = 0.
            sparse_initial_solution = project(exact_solution_expression, V)
            dense_solution = DenseFunction(*sparse_initial_solution.vector().array().shape)
            dense_solution.vector()[:] = sparse_initial_solution.vector().array().reshape((-1, 1))
            dense_solution_array = dense_solution.vector()
            dense_solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            return dense_solution
        
        # Define custom monitor to plot the solution
        def monitor(self, t, solution, solution_dot):
            self._solution_from_dense_to_sparse(solution, u)
            self._solution_from_dense_to_sparse(solution_dot, u_dot)
            plt.subplot(1, 2, 1).clear()
            plot(u, title="u at t = " + str(t))
            plt.subplot(1, 2, 2).clear()
            plot(u_dot, title="u_dot at t = " + str(t))
            plt.show(block=False)
            plt.pause(DOLFIN_EPS)
            
        def _solution_from_dense_to_sparse(self, solution, u):
            solution_array = asarray(solution.vector()).reshape(-1)
            solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
            u.vector().zero()
            u.vector().add_local(solution_array)
            u.vector().apply("")
            solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
    
    for integrator_type in ("beuler", "ida"):
        # Solve the time dependent problem
        dense_problem_wrapper = DenseProblemWrapper()
        dense_shape = sparse_solution.vector().array().shape
        (dense_solution, dense_solution_dot) = (DenseFunction(*dense_shape), DenseFunction(*dense_shape))
        dense_solver = DenseTimeStepping(dense_problem_wrapper, dense_solution, dense_solution_dot)
        dense_solver_parameters = {
            "initial_time": 0.0,
            "time_step_size": dt,
            "final_time": T,
            "integrator_type": integrator_type,
            "problem_type": "nonlinear",
            "monitor": dense_problem_wrapper.monitor,
            "report": True
        }
        if integrator_type != "ida":
            dense_solver_parameters.update({
                "nonlinear_solver": {
                    "maximum_iterations": 20,
                    "report": True
                }
            })
        dense_solver.set_parameters(dense_solver_parameters)
        all_dense_solutions_time, all_dense_solutions, all_dense_solutions_dot = dense_solver.solve()
        assert len(all_dense_solutions_time) == int(T/dt + 1)
        assert len(all_dense_solutions) == int(T/dt + 1)
        assert len(all_dense_solutions_dot) == int(T/dt + 1)
        dense_solution_array = dense_solution.vector()
        dense_solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = dense_solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
        dense_solution_dot_array = dense_solution_dot.vector()
        dense_solution_dot_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = dense_solution_dot_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
        
        # Compute the error
        dense_error = DenseFunction(*exact_solution.vector().array().shape)
        dense_error.vector()[:] = exact_solution.vector().array().reshape((-1, 1))
        dense_error.vector()[:] -= dense_solution_array
        dense_error_norm = dense_error.vector().T*(X.array()*dense_error.vector())
        assert dense_error_norm.shape == (1, 1)
        dense_error_norm = dense_error_norm[0, 0]
        dense_error_dot = DenseFunction(*exact_solution_dot.vector().array().shape)
        dense_error_dot.vector()[:] = exact_solution_dot.vector().array().reshape((-1, 1))
        dense_error_dot.vector()[:] -= dense_solution_dot_array
        dense_error_dot_norm = dense_error_dot.vector().T*(X.array()*dense_error_dot.vector())
        assert dense_error_dot_norm.shape == (1, 1)
        dense_error_dot_norm = dense_error_dot_norm[0, 0]
        print "DenseTimeStepping error:", dense_error_norm, dense_error_dot_norm
        assert isclose(dense_error_norm, 0., atol=1.e-4)
        assert isclose(dense_error_dot_norm, 0., atol=1.e-4)
else:
    print "DenseTimeStepping error: skipped in parallel"
