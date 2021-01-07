# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from numpy import isclose
from dolfin import (assemble, Constant, DirichletBC, div, DOLFIN_EPS, dx, FiniteElement, FunctionSpace, grad, inner,
                    MeshFunction, MixedElement, split, SubDomain, TestFunction, TrialFunction, UnitSquareMesh,
                    VectorElement)

"""
Computation of the inf-sup constant
of a Stokes problem on a square
"""


# ~~~ Sparse case ~~~ #
def _test_eigen_solver_sparse(callback_type):
    from rbnics.backends.dolfin import EigenSolver

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
    lhs = inner(grad(u), grad(v)) * dx - div(v) * p * dx - div(u) * q * dx
    rhs = - inner(p, q) * dx

    # Define boundary condition
    bc = [DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, 1)]

    # Define eigensolver depending on callback type
    assert callback_type in ("form callbacks", "tensor callbacks")
    if callback_type == "form callbacks":
        solver = EigenSolver(W, lhs, rhs, bc)
    elif callback_type == "tensor callbacks":
        LHS = assemble(lhs)
        RHS = assemble(rhs)
        solver = EigenSolver(W, LHS, RHS, bc)

    # Solve the eigenproblem
    solver.set_parameters({
        "linear_solver": "mumps",
        "problem_type": "gen_non_hermitian",
        "spectrum": "target real",
        "spectral_transform": "shift-and-invert",
        "spectral_shift": 1.e-5
    })
    solver.solve(1)
    r, c = solver.get_eigenvalue(0)
    assert abs(c) < 1.e-10
    assert r > 0., "r = " + str(r) + " is not positive"
    print("Sparse inf-sup constant: ", sqrt(r))
    return (sqrt(r), solver.condensed_A, solver.condensed_B)


# ~~~ Dense case ~~~ #
def _test_eigen_solver_dense(sparse_LHS, sparse_RHS):
    from rbnics.backends.online.numpy import EigenSolver, Matrix

    # Extract constrained matrices from sparse eigensolver
    LHS = Matrix(*sparse_LHS.array().shape)
    RHS = Matrix(*sparse_RHS.array().shape)
    LHS[:, :] = sparse_LHS.array()
    RHS[:, :] = sparse_RHS.array()

    # Solve the eigenproblem
    solver = EigenSolver(None, LHS, RHS)
    solver.set_parameters({
        "problem_type": "gen_non_hermitian",
        "spectrum": "smallest real",
    })
    solver.solve(1)
    r, c = solver.get_eigenvalue(0)
    assert abs(c) < 1.e-10
    assert r > 0., "r = " + str(r) + " is not positive"
    print("Dense inf-sup constant: ", sqrt(r))
    return (sqrt(r), LHS, RHS)


# ~~~ Test function ~~~ #
def test_eigen_solver():
    sqrt_r_exact = 0.6051627263949135
    (sqrt_r_sparse_tensor_callbacks, sparse_LHS, sparse_RHS) = _test_eigen_solver_sparse("tensor callbacks")
    assert isclose(sqrt_r_sparse_tensor_callbacks, sqrt_r_exact)
    (sqrt_r_sparse_form_callbacks, _, _) = _test_eigen_solver_sparse("form callbacks")
    assert isclose(sqrt_r_sparse_form_callbacks, sqrt_r_exact)
    if sparse_LHS.mpi_comm().size == 1:  # dense solver is not partitioned
        (sqrt_r_dense, _, _) = _test_eigen_solver_dense(sparse_LHS, sparse_RHS)
        assert isclose(sqrt_r_dense, sqrt_r_exact)
        # Compute the error
        sqrt_r_dense_error = abs(sqrt_r_sparse_tensor_callbacks - sqrt_r_dense)
        print("Dense error:", sqrt_r_dense_error)
        assert isclose(sqrt_r_dense_error, 0., atol=1.e-5)
