# Copyright (C) 2015-2018 by the RBniCS authors
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

from math import sqrt
from numpy import isclose
from dolfin import assemble, Constant, DirichletBC, div, DOLFIN_EPS, dx, FiniteElement, FunctionSpace, grad, inner, MeshFunction, MixedElement, split, SubDomain, TestFunction, TrialFunction, UnitSquareMesh, VectorElement
from rbnics.backends.dolfin import EigenSolver as SparseEigenSolver
from rbnics.backends.online.numpy import EigenSolver as DenseEigenSolver, Matrix as DenseMatrix

"""
Computation of the inf-sup constant
of a Stokes problem on a square
"""

# ~~~ Sparse case ~~~ #
def _test_eigen_solver_sparse(callback_type):
    # Define mesh
    mesh = UnitSquareMesh(10, 10)

    # Define function space
    V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_element = MixedElement(V_element, Q_element)
    W = FunctionSpace(mesh, W_element)

    # Create boundaries
    class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[1] < 0 + DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS)
        
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    wall = Wall()
    wall.mark(boundaries, 1)

    # Define variational problem
    vq = TestFunction(W)
    (v, q) = split(vq)
    up = TrialFunction(W)
    (u, p) = split(up)
    lhs = inner(grad(u), grad(v))*dx - div(v)*p*dx - div(u)*q*dx
    rhs = - inner(p, q)*dx

    # Define boundary condition
    bc = [DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, 1)]

    # Define eigensolver depending on callback type
    assert callback_type in ("form callbacks", "tensor callbacks")
    if callback_type == "form callbacks":
        sparse_solver = SparseEigenSolver(W, lhs, rhs, bc)
    elif callback_type == "tensor callbacks":
        LHS = assemble(lhs)
        RHS = assemble(rhs)
        sparse_solver = SparseEigenSolver(W, LHS, RHS, bc)

    # Solve the eigenproblem
    sparse_solver.set_parameters({
        "problem_type": "gen_non_hermitian",
        "spectrum": "target real",
        "spectral_transform": "shift-and-invert",
        "spectral_shift": 1.e-5
    })
    sparse_solver.solve(1)
    r, c = sparse_solver.get_eigenvalue(0)
    assert abs(c) < 1.e-10
    assert r > 0., "r = " + str(r) + " is not positive"
    print("Sparse inf-sup constant: ", sqrt(r))
    return (sqrt(r), sparse_solver.condensed_A, sparse_solver.condensed_B)

# ~~~ Dense case ~~~ #
def _test_eigen_solver_dense(sparse_LHS, sparse_RHS):
    # Extract constrained matrices from sparse eigensolver
    dense_LHS_array = sparse_LHS.array()
    dense_RHS_array = sparse_RHS.array()
    
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
    print("Dense inf-sup constant: ", sqrt(dense_r))
    return (sqrt(dense_r), dense_LHS_array, dense_RHS_array)
    
# ~~~ Test function ~~~ #
def test_eigen_solver():
    sqrt_r_exact = 0.6051627263949135
    (sqrt_r_sparse_tensor_callbacks, sparse_LHS, sparse_RHS) = _test_eigen_solver_sparse("tensor callbacks")
    assert isclose(sqrt_r_sparse_tensor_callbacks, sqrt_r_exact)
    (sqrt_r_sparse_form_callbacks, _, _) = _test_eigen_solver_sparse("form callbacks")
    assert isclose(sqrt_r_sparse_form_callbacks, sqrt_r_exact)
    if sparse_LHS.mpi_comm().size == 1: # dense solver is not partitioned
        (sqrt_r_dense, _, _) = _test_eigen_solver_dense(sparse_LHS, sparse_RHS)
        assert isclose(sqrt_r_dense, sqrt_r_exact)
        # Compute the error
        sqrt_r_dense_error = abs(sqrt_r_sparse_tensor_callbacks - sqrt_r_dense)
        print("DenseEigenSolver error:", sqrt_r_dense_error)
        assert isclose(sqrt_r_dense_error, 0., atol=1.e-5)
