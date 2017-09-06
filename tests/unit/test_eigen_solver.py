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

from numpy import isclose
from dolfin import *
from math import sqrt
from rbnics.backends.dolfin import EigenSolver as SparseEigenSolver
from rbnics.backends.online.numpy import EigenSolver as DenseEigenSolver, Matrix as DenseMatrix

"""
Computation of the inf-sup constant
of a Stokes problem on a square
"""

# Create mesh and define function space
mesh = UnitSquareMesh(10, 10)
V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element)
W = FunctionSpace(mesh, W_element)

# Create boundaries
class Wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < 0 + DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS)
    
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
wall = Wall()
wall.mark(boundaries, 1)

# Define variational problem
vq  = TestFunction(W)
(v, q) = split(vq)
up = TrialFunction(W)
(u, p) = split(up)
lhs = (   inner(grad(u), grad(v))*dx
          - div(v)*p*dx
          - div(u)*q*dx
        )
rhs =   - inner(p, q)*dx

# Assemble matrix and vector
LHS = assemble(lhs)
RHS = assemble(rhs)

# ~~~ Sparse case ~~~ #
# Define boundary condition
bc = [DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, 1)]

# Solve the eigenproblem
sparse_solver = SparseEigenSolver(W, LHS, RHS, bc)
sparse_solver.solve(1)
sparse_solver.set_parameters({
    "problem_type": "gen_non_hermitian", 
    "spectrum": "smallest real",
    "spectral_transform": "shift-and-invert",
    "spectral_shift": 1.e-5
})
sparse_solver.solve(1)
r, c = sparse_solver.get_eigenvalue(0)
assert abs(c) < 1.e-10
assert r > 0., "r = " + str(r) + " is not positive"
print "Sparse inf-sup constant: ", sqrt(r)

# ~~~ Dense case ~~~ #
if mesh.mpi_comm().size == 1: # dense solver is not partitioned    
    # Extract constrained matrices from sparse eigensolver
    dense_LHS_array = sparse_solver.condensed_A.array()
    dense_RHS_array = sparse_solver.condensed_B.array()
    
    # Convert to dense format
    dense_LHS = DenseMatrix(*dense_LHS_array.shape)
    dense_RHS = DenseMatrix(*dense_RHS_array.shape)
    dense_LHS[:, :] = dense_LHS_array
    dense_RHS[:, :] = dense_RHS_array
    
    # Solve the eigenproblem
    dense_solver = DenseEigenSolver(None, dense_LHS, dense_RHS)
    dense_solver.set_parameters({
        "problem_type": "gen_non_hermitian", 
        "spectrum": "smallest real",
    })
    dense_solver.solve(1)
    dense_r, dense_c = dense_solver.get_eigenvalue(0)
    assert abs(dense_c) < 1.e-10
    assert dense_r > 0., "dense_r = " + str(dense_r) + " is not positive"
    print "Dense inf-sup constant: ", sqrt(dense_r)
    
    # Compute the error
    dense_sqrt_r_error = abs(sqrt(r) - sqrt(dense_r))
    print "DenseEigenSolver error:", dense_sqrt_r_error
    assert isclose(dense_sqrt_r_error, 0., atol=1.e-5)
else:
    print "DenseEigenSolver error: skipped in parallel"
    

