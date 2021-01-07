# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from mpi4py import MPI
from logging import DEBUG, getLogger
from numpy import array, isclose, nonzero, sort
from dolfin import (assemble, dx, Expression, FiniteElement, FunctionSpace, inner, MixedElement, Point, project,
                    split, TestFunction, TrialFunction, UnitIntervalMesh, UnitSquareMesh, Vector, VectorElement)
try:
    from mshr import generate_mesh, Rectangle
except ImportError:
    has_mshr = False
else:
    has_mshr = True
from rbnics.backends.dolfin import ReducedMesh
from rbnics.backends.dolfin.wrapping import (
    evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs, evaluate_sparse_vector_at_dofs)
from rbnics.utils.test import enable_logging

# Logger
test_logger = getLogger("tests/unit/test_reduced_mesh.py")
enable_reduced_mesh_logging = enable_logging({test_logger: DEBUG})


# Meshes
def structured_mesh_1d():
    return UnitIntervalMesh(24)


def structured_mesh_2d():
    return UnitSquareMesh(3, 3)


if has_mshr:
    def unstructured_mesh_2d():
        domain = Rectangle(Point(0., 0.), Point(1., 1.))
        return generate_mesh(domain, 5)

    generate_meshes = pytest.mark.parametrize("mesh", [
        structured_mesh_1d(), structured_mesh_2d(), unstructured_mesh_2d()])
else:
    generate_meshes = pytest.mark.parametrize("mesh", [
        structured_mesh_1d(), structured_mesh_2d()])


# Helper functions
def nonzero_values(function):
    serialized_vector = Vector(MPI.COMM_SELF)
    function.vector().gather(serialized_vector, array(range(function.function_space().dim()), "intc"))
    indices = nonzero(serialized_vector.get_local())
    return sort(serialized_vector.get_local()[indices])


# ~~~ Elliptic case ~~~ #
def EllipticFunctionSpace(mesh):
    return FunctionSpace(mesh, "CG", 2)


# === Matrix computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_elliptic_matrix(mesh, tempdir):
    test_reduced_mesh_save_elliptic_matrix(mesh, tempdir)
    test_reduced_mesh_load_elliptic_matrix(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_elliptic_matrix(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Elliptic case, matrix, offline computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, V))
    dofs = [(1, 2), (11, 12), (48, 12), (41, 41)]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_elliptic_matrix")

    _test_reduced_mesh_elliptic_matrix(V, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_elliptic_matrix(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Elliptic case, matrix, online computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, V))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_elliptic_matrix")

    _test_reduced_mesh_elliptic_matrix(V, reduced_mesh)


def _test_reduced_mesh_elliptic_matrix(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = reduced_mesh.get_dofs_list()
    reduced_dofs = reduced_mesh.get_reduced_dofs_list()

    u = TrialFunction(V)
    v = TestFunction(V)

    trial = 1
    test = 0
    u_N = TrialFunction(reduced_V[trial])
    v_N = TestFunction(reduced_V[test])

    A = assemble((u.dx(0) * v + u * v) * dx)
    A_N = assemble((u_N.dx(0) * v_N + u_N * v_N) * dx)

    A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
    A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

    test_logger.log(DEBUG, "A at dofs:")
    test_logger.log(DEBUG, str(A_dofs))
    test_logger.log(DEBUG, "A_N at reduced dofs:")
    test_logger.log(DEBUG, str(A_N_reduced_dofs))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(A_dofs - A_N_reduced_dofs))

    assert isclose(A_dofs, A_N_reduced_dofs).all()


# === Vector computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_elliptic_vector(mesh, tempdir):
    test_reduced_mesh_save_elliptic_vector(mesh, tempdir)
    test_reduced_mesh_load_elliptic_vector(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_elliptic_vector(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Elliptic case, vector, offline computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(1, ), (11, ), (48, ), (41, )]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_elliptic_vector")

    _test_reduced_mesh_elliptic_vector(V, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_elliptic_vector(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Elliptic case, vector, online computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_elliptic_vector")

    _test_reduced_mesh_elliptic_vector(V, reduced_mesh)


def _test_reduced_mesh_elliptic_vector(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = reduced_mesh.get_dofs_list()
    reduced_dofs = reduced_mesh.get_reduced_dofs_list()

    v = TestFunction(V)

    test = 0
    v_N = TestFunction(reduced_V[test])

    b = assemble(v * dx)
    b_N = assemble(v_N * dx)

    b_dofs = evaluate_sparse_vector_at_dofs(b, dofs)
    b_N_reduced_dofs = evaluate_sparse_vector_at_dofs(b_N, reduced_dofs)

    test_logger.log(DEBUG, "b at dofs:")
    test_logger.log(DEBUG, str(b_dofs))
    test_logger.log(DEBUG, "b_N at reduced dofs:")
    test_logger.log(DEBUG, str(b_N_reduced_dofs))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(b_dofs - b_N_reduced_dofs))

    assert isclose(b_dofs, b_N_reduced_dofs).all()


# === Function computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_elliptic_function(mesh, tempdir):
    test_reduced_mesh_save_elliptic_function(mesh, tempdir)
    test_reduced_mesh_load_elliptic_function(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_elliptic_function(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Elliptic case, function, offline computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(1, ), (11, ), (48, ), (41, )]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_elliptic_function")

    _test_reduced_mesh_elliptic_function(V, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_elliptic_function(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Elliptic case, function, online computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_elliptic_function")

    _test_reduced_mesh_elliptic_function(V, reduced_mesh)


def _test_reduced_mesh_elliptic_function(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = [d[0] for d in reduced_mesh.get_dofs_list()]  # convert from 1-tuple to int
    reduced_dofs = [d[0] for d in reduced_mesh.get_reduced_dofs_list()]  # convert from 1-tuple to int

    mesh_dim = V.mesh().geometry().dim()
    assert mesh_dim in (1, 2)
    if mesh_dim == 1:
        e = Expression("1+x[0]", element=V.ufl_element())
    else:
        e = Expression("(1+x[0])*(1+x[1])", element=V.ufl_element())

    f = project(e, V)
    f_N = project(e, reduced_V[0])

    f_dofs = evaluate_sparse_function_at_dofs(f, dofs, V, dofs)
    f_reduced_dofs = evaluate_sparse_function_at_dofs(f, dofs, reduced_V[0], reduced_dofs)
    f_N_reduced_dofs = evaluate_sparse_function_at_dofs(f_N, reduced_dofs, reduced_V[0], reduced_dofs)

    test_logger.log(DEBUG, "f at dofs:")
    test_logger.log(DEBUG, str(nonzero_values(f_dofs)))
    test_logger.log(DEBUG, "f at reduced dofs:")
    test_logger.log(DEBUG, str(nonzero_values(f_reduced_dofs)))
    test_logger.log(DEBUG, "f_N at reduced dofs:")
    test_logger.log(DEBUG, str(nonzero_values(f_N_reduced_dofs)))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(nonzero_values(f_dofs) - nonzero_values(f_reduced_dofs)))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(f_reduced_dofs.vector().get_local() - f_N_reduced_dofs.vector().get_local()))

    assert isclose(nonzero_values(f_dofs), nonzero_values(f_reduced_dofs)).all()
    assert isclose(f_reduced_dofs.vector().get_local(), f_N_reduced_dofs.vector().get_local()).all()


# ~~~ Mixed case ~~~ #
def MixedFunctionSpace(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return FunctionSpace(mesh, element)


# === Matrix computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_mixed_matrix(mesh, tempdir):
    test_reduced_mesh_save_mixed_matrix(mesh, tempdir)
    test_reduced_mesh_load_mixed_matrix(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_mixed_matrix(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Mixed case, matrix, offline computation ***")
    V = MixedFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, V))
    dofs = [(1, 2), (31, 33), (48, 12), (42, 42)]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_mixed_matrix")

    _test_reduced_mesh_mixed_matrix(V, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_mixed_matrix(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Mixed case, matrix, online computation ***")
    V = MixedFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, V))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_mixed_matrix")

    _test_reduced_mesh_mixed_matrix(V, reduced_mesh)


def _test_reduced_mesh_mixed_matrix(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = reduced_mesh.get_dofs_list()
    reduced_dofs = reduced_mesh.get_reduced_dofs_list()

    u = TrialFunction(V)
    v = TestFunction(V)
    (u_0, u_1) = split(u)
    (v_0, v_1) = split(v)

    trial = 1
    test = 0
    u_N = TrialFunction(reduced_V[trial])
    v_N = TestFunction(reduced_V[test])
    (u_N_0, u_N_1) = split(u_N)
    (v_N_0, v_N_1) = split(v_N)

    A = assemble(
        u_0[0] * v_0[0] * dx + u_0[0] * v_0[1] * dx + u_0[1] * v_0[0] * dx + u_0[1] * v_0[1] * dx
        + u_1 * v_1 * dx + u_0[0] * v_1 * dx + u_1 * v_0[1] * dx)
    A_N = assemble(
        u_N_0[0] * v_N_0[0] * dx + u_N_0[0] * v_N_0[1] * dx + u_N_0[1] * v_N_0[0] * dx + u_N_0[1] * v_N_0[1] * dx
        + u_N_1 * v_N_1 * dx + u_N_0[0] * v_N_1 * dx + u_N_1 * v_N_0[1] * dx)

    A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
    A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

    test_logger.log(DEBUG, "A at dofs:")
    test_logger.log(DEBUG, str(A_dofs))
    test_logger.log(DEBUG, "A_N at reduced dofs:")
    test_logger.log(DEBUG, str(A_N_reduced_dofs))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(A_dofs - A_N_reduced_dofs))

    assert isclose(A_dofs, A_N_reduced_dofs).all()


# === Vector computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_mixed_vector(mesh, tempdir):
    test_reduced_mesh_save_mixed_vector(mesh, tempdir)
    test_reduced_mesh_load_mixed_vector(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_mixed_vector(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Mixed case, vector, offline computation ***")
    V = MixedFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(2, ), (33, ), (48, ), (42, )]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_mixed_vector")

    _test_reduced_mesh_mixed_vector(V, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_mixed_vector(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Mixed case, vector, online computation ***")
    V = MixedFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_mixed_vector")

    _test_reduced_mesh_mixed_vector(V, reduced_mesh)


def _test_reduced_mesh_mixed_vector(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = reduced_mesh.get_dofs_list()
    reduced_dofs = reduced_mesh.get_reduced_dofs_list()

    v = TestFunction(V)
    (v_0, v_1) = split(v)

    test = 0
    v_N = TestFunction(reduced_V[test])
    (v_N_0, v_N_1) = split(v_N)

    b = assemble(v_0[0] * dx + v_0[1] * dx + v_1 * dx)
    b_N = assemble(v_N_0[0] * dx + v_N_0[1] * dx + v_N_1 * dx)

    b_dofs = evaluate_sparse_vector_at_dofs(b, dofs)
    b_N_reduced_dofs = evaluate_sparse_vector_at_dofs(b_N, reduced_dofs)

    test_logger.log(DEBUG, "b at dofs:")
    test_logger.log(DEBUG, str(b_dofs))
    test_logger.log(DEBUG, "b_N at reduced dofs:")
    test_logger.log(DEBUG, str(b_N_reduced_dofs))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(b_dofs - b_N_reduced_dofs))

    assert isclose(b_dofs, b_N_reduced_dofs).all()


# ~~~ Collapsed case ~~~ #
def CollapsedFunctionSpaces(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    U = FunctionSpace(mesh, element)
    V = U.sub(0).collapse()
    return (V, U)


# === Matrix computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_collapsed_matrix(mesh, tempdir):
    test_reduced_mesh_save_collapsed_matrix(mesh, tempdir)
    test_reduced_mesh_load_collapsed_matrix(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_collapsed_matrix(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Collapsed case, matrix, offline computation ***")
    (V, U) = CollapsedFunctionSpaces(mesh)
    reduced_mesh = ReducedMesh((V, U))
    dofs = [(2, 1), (48, 33), (40, 12), (31, 39)]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_collapsed_matrix")

    _test_reduced_mesh_collapsed_matrix(V, U, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_collapsed_matrix(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Collapsed case, matrix, online computation ***")
    (V, U) = CollapsedFunctionSpaces(mesh)
    reduced_mesh = ReducedMesh((V, U))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_collapsed_matrix")

    _test_reduced_mesh_collapsed_matrix(V, U, reduced_mesh)


def _test_reduced_mesh_collapsed_matrix(V, U, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = reduced_mesh.get_dofs_list()
    reduced_dofs = reduced_mesh.get_reduced_dofs_list()

    u = TrialFunction(U)
    (u_0, u_1) = split(u)
    v = TestFunction(V)

    trial = 1
    test = 0
    u_N = TrialFunction(reduced_V[trial])
    v_N = TestFunction(reduced_V[test])
    (u_N_0, u_N_1) = split(u_N)

    A = assemble(inner(u_0, v) * dx + u_1 * v[0] * dx)
    A_N = assemble(inner(u_N_0, v_N) * dx + u_N_1 * v_N[0] * dx)

    A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
    A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

    test_logger.log(DEBUG, "A at dofs:")
    test_logger.log(DEBUG, str(A_dofs))
    test_logger.log(DEBUG, "A_N at reduced dofs:")
    test_logger.log(DEBUG, str(A_N_reduced_dofs))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(A_dofs - A_N_reduced_dofs))

    assert isclose(A_dofs, A_N_reduced_dofs).all()


# === Vector computation === #
@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_io_collapsed_vector(mesh, tempdir):
    test_reduced_mesh_save_collapsed_vector(mesh, tempdir)
    test_reduced_mesh_load_collapsed_vector(mesh, tempdir)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_save_collapsed_vector(mesh, save_tempdir):
    test_logger.log(DEBUG, "*** Collapsed case, vector, offline computation ***")
    (V, _) = CollapsedFunctionSpaces(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(2, ), (48, ), (40, ), (11, )]

    for pair in dofs:
        test_logger.log(DEBUG, "Adding " + str(pair))
        reduced_mesh.append(pair)

        reduced_mesh.save(save_tempdir, "test_reduced_mesh_collapsed_vector")

    _test_reduced_mesh_collapsed_vector(V, reduced_mesh)


@generate_meshes
@enable_reduced_mesh_logging
def test_reduced_mesh_load_collapsed_vector(mesh, load_tempdir):
    test_logger.log(DEBUG, "*** Collapsed case, vector, online computation ***")
    (V, _) = CollapsedFunctionSpaces(mesh)
    reduced_mesh = ReducedMesh((V, ))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_collapsed_vector")

    _test_reduced_mesh_collapsed_vector(V, reduced_mesh)


def _test_reduced_mesh_collapsed_vector(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = reduced_mesh.get_dofs_list()
    reduced_dofs = reduced_mesh.get_reduced_dofs_list()

    v = TestFunction(V)

    test = 0
    v_N = TestFunction(reduced_V[test])

    b = assemble(v[0] * dx + v[1] * dx)
    b_N = assemble(v_N[0] * dx + v_N[1] * dx)

    b_dofs = evaluate_sparse_vector_at_dofs(b, dofs)
    b_N_reduced_dofs = evaluate_sparse_vector_at_dofs(b_N, reduced_dofs)

    test_logger.log(DEBUG, "b at dofs:")
    test_logger.log(DEBUG, str(b_dofs))
    test_logger.log(DEBUG, "b_N at reduced dofs:")
    test_logger.log(DEBUG, str(b_N_reduced_dofs))
    test_logger.log(DEBUG, "Error:")
    test_logger.log(DEBUG, str(b_dofs - b_N_reduced_dofs))
