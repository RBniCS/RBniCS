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

import pytest
from numpy import allclose, array, nonzero, sort
from dolfin import assemble, dx, Expression, FiniteElement, FunctionSpace, has_pybind11, inner, MixedElement, mpi_comm_self, Point, project, split, TestFunction, TrialFunction, UnitSquareMesh, Vector, VectorElement
if has_pybind11():
    from dolfin.cpp.log import log, LogLevel, set_log_level
    PROGRESS = LogLevel.PROGRESS
else:
    from dolfin import log, PROGRESS, set_log_level
set_log_level(PROGRESS)
try:
    from mshr import generate_mesh, Rectangle
except ImportError:
    has_mshr = False
else:
    has_mshr = True
if has_pybind11():
    has_mshr = False # TODO mshr still uses swig wrapping
from rbnics.backends.dolfin import ReducedMesh
from rbnics.backends.dolfin.wrapping import evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs, evaluate_sparse_vector_at_dofs
from rbnics.backends.online import OnlineMatrix, OnlineVector

# Meshes
def structured_mesh():
    return UnitSquareMesh(3, 3)
    
if has_mshr:
    def unstructured_mesh():
        domain = Rectangle(Point(0., 0.), Point(1., 1.))
        return generate_mesh(domain, 5)
        
    generate_meshes = pytest.mark.parametrize("mesh", [structured_mesh(), unstructured_mesh()])
else:
    generate_meshes = pytest.mark.parametrize("mesh", [structured_mesh()])

# Helper functions
def nonzero_values(function):
    serialized_vector = Vector(mpi_comm_self())
    function.vector().gather(serialized_vector, array(range(function.function_space().dim()), "intc"))
    indices = nonzero(serialized_vector.get_local())
    return sort(serialized_vector.get_local()[indices])
    
def isclose(a, b):
    assert type(a) is type(b)
    if isinstance(a, (OnlineMatrix.Type(), OnlineVector.Type())):
        return allclose(a.content, b.content)
    else:
        return allclose(a, b)

# ~~~ Elliptic case ~~~ #
def EllipticFunctionSpace(mesh):
    return FunctionSpace(mesh, "CG", 2)

# === Matrix computation === #
@generate_meshes
def test_reduced_mesh_io_elliptic_matrix(mesh, tempdir):
    test_reduced_mesh_save_elliptic_matrix(mesh, tempdir)
    test_reduced_mesh_load_elliptic_matrix(mesh, tempdir)

@generate_meshes
def test_reduced_mesh_save_elliptic_matrix(mesh, save_tempdir):
    log(PROGRESS, "*** Elliptic case, matrix, offline computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, V))
    dofs = [(1, 2), (11, 12), (48, 12), (41, 41)]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_elliptic_matrix")
    
    _test_reduced_mesh_elliptic_matrix(V, reduced_mesh)
    
@generate_meshes
def test_reduced_mesh_load_elliptic_matrix(mesh, load_tempdir):
    log(PROGRESS, "*** Elliptic case, matrix, online computation ***")
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

    A = assemble((u.dx(0)*v + u*v)*dx)
    A_N = assemble((u_N.dx(0)*v_N + u_N*v_N)*dx)

    A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
    A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

    log(PROGRESS, "A at dofs:\n" + str(A_dofs))
    log(PROGRESS, "A_N at reduced dofs:\n" + str(A_N_reduced_dofs))
    log(PROGRESS, "Error:\n" + str(A_dofs - A_N_reduced_dofs))
    
    assert isclose(A_dofs, A_N_reduced_dofs)
    
# === Vector computation === #
@generate_meshes
def test_reduced_mesh_io_elliptic_vector(mesh, tempdir):
    test_reduced_mesh_save_elliptic_vector(mesh, tempdir)
    test_reduced_mesh_load_elliptic_vector(mesh, tempdir)
    
@generate_meshes
def test_reduced_mesh_save_elliptic_vector(mesh, save_tempdir):
    log(PROGRESS, "*** Elliptic case, vector, offline computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(1, ), (11, ), (48, ), (41, )]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_elliptic_vector")
        
    _test_reduced_mesh_elliptic_vector(V, reduced_mesh)

@generate_meshes
def test_reduced_mesh_load_elliptic_vector(mesh, load_tempdir):
    log(PROGRESS, "*** Elliptic case, vector, online computation ***")
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

    b = assemble(v*dx)
    b_N = assemble(v_N*dx)
    
    b_dofs = evaluate_sparse_vector_at_dofs(b, dofs)
    b_N_reduced_dofs = evaluate_sparse_vector_at_dofs(b_N, reduced_dofs)
    
    log(PROGRESS, "b at dofs:\n" + str(b_dofs))
    log(PROGRESS, "b_N at reduced dofs:\n" + str(b_N_reduced_dofs))
    log(PROGRESS, "Error:\n" + str(b_dofs - b_N_reduced_dofs))
    
    assert isclose(b_dofs, b_N_reduced_dofs)

# === Function computation === #
@generate_meshes
def test_reduced_mesh_io_elliptic_function(mesh, tempdir):
    test_reduced_mesh_save_elliptic_function(mesh, tempdir)
    test_reduced_mesh_load_elliptic_function(mesh, tempdir)
    
@generate_meshes
def test_reduced_mesh_save_elliptic_function(mesh, save_tempdir):
    log(PROGRESS, "*** Elliptic case, function, offline computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(1, ), (11, ), (48, ), (41, )]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_elliptic_function")
        
    _test_reduced_mesh_elliptic_function(V, reduced_mesh)

@generate_meshes
def test_reduced_mesh_load_elliptic_function(mesh, load_tempdir):
    log(PROGRESS, "*** Elliptic case, function, online computation ***")
    V = EllipticFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    reduced_mesh.load(load_tempdir, "test_reduced_mesh_elliptic_function")
    
    _test_reduced_mesh_elliptic_function(V, reduced_mesh)

def _test_reduced_mesh_elliptic_function(V, reduced_mesh):
    reduced_V = reduced_mesh.get_reduced_function_spaces()
    dofs = [d[0] for d in reduced_mesh.get_dofs_list()] # convert from 1-tuple to int
    reduced_dofs = [d[0] for d in reduced_mesh.get_reduced_dofs_list()] # convert from 1-tuple to int
    
    e = Expression("(1+x[0])*(1+x[1])", element=V.ufl_element())
    
    f = project(e, V)
    f_N = project(e, reduced_V[0])
    
    f_dofs = evaluate_sparse_function_at_dofs(f, dofs, V, dofs)
    f_reduced_dofs = evaluate_sparse_function_at_dofs(f, dofs, reduced_V[0], reduced_dofs)
    f_N_reduced_dofs = evaluate_sparse_function_at_dofs(f_N, reduced_dofs, reduced_V[0], reduced_dofs)
    
    log(PROGRESS, "f at dofs:\n" + str(nonzero_values(f_dofs)))
    log(PROGRESS, "f at reduced dofs:\n" + str(nonzero_values(f_reduced_dofs)))
    log(PROGRESS, "f_N at reduced dofs:\n" + str(nonzero_values(f_N_reduced_dofs)))
    log(PROGRESS, "Error:\n" + str(nonzero_values(f_dofs) - nonzero_values(f_reduced_dofs)))
    log(PROGRESS, "Error:\n" + str(f_reduced_dofs.vector().get_local() - f_N_reduced_dofs.vector().get_local()))
    
    assert isclose(nonzero_values(f_dofs), nonzero_values(f_reduced_dofs))
    assert isclose(f_reduced_dofs.vector().get_local(), f_N_reduced_dofs.vector().get_local())

# ~~~ Mixed case ~~~ #
def MixedFunctionSpace(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return FunctionSpace(mesh, element)

# === Matrix computation === #
@generate_meshes
def test_reduced_mesh_io_mixed_matrix(mesh, tempdir):
    test_reduced_mesh_save_mixed_matrix(mesh, tempdir)
    test_reduced_mesh_load_mixed_matrix(mesh, tempdir)

@generate_meshes
def test_reduced_mesh_save_mixed_matrix(mesh, save_tempdir):
    log(PROGRESS, "*** Mixed case, matrix, offline computation ***")
    V = MixedFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, V))
    dofs = [(1, 2), (31, 33), (48, 12), (42, 42)]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_mixed_matrix")
    
    _test_reduced_mesh_mixed_matrix(V, reduced_mesh)
    
@generate_meshes
def test_reduced_mesh_load_mixed_matrix(mesh, load_tempdir):
    log(PROGRESS, "*** Mixed case, matrix, online computation ***")
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

    A = assemble(u_0[0]*v_0[0]*dx + u_0[0]*v_0[1]*dx + u_0[1]*v_0[0]*dx + u_0[1]*v_0[1]*dx + u_1*v_1*dx + u_0[0]*v_1*dx + u_1*v_0[1]*dx)
    A_N = assemble(u_N_0[0]*v_N_0[0]*dx + u_N_0[0]*v_N_0[1]*dx + u_N_0[1]*v_N_0[0]*dx + u_N_0[1]*v_N_0[1]*dx + u_N_1*v_N_1*dx + u_N_0[0]*v_N_1*dx + u_N_1*v_N_0[1]*dx)

    A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
    A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

    log(PROGRESS, "A at dofs:\n" + str(A_dofs))
    log(PROGRESS, "A_N at reduced dofs:\n" + str(A_N_reduced_dofs))
    log(PROGRESS, "Error:\n" + str(A_dofs - A_N_reduced_dofs))
    
    assert isclose(A_dofs, A_N_reduced_dofs)
    
# === Vector computation === #
@generate_meshes
def test_reduced_mesh_io_mixed_vector(mesh, tempdir):
    test_reduced_mesh_save_mixed_vector(mesh, tempdir)
    test_reduced_mesh_load_mixed_vector(mesh, tempdir)
    
@generate_meshes
def test_reduced_mesh_save_mixed_vector(mesh, save_tempdir):
    log(PROGRESS, "*** Mixed case, vector, offline computation ***")
    V = MixedFunctionSpace(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(2, ), (33, ), (48, ), (42, )]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_mixed_vector")
        
    _test_reduced_mesh_mixed_vector(V, reduced_mesh)

@generate_meshes
def test_reduced_mesh_load_mixed_vector(mesh, load_tempdir):
    log(PROGRESS, "*** Mixed case, vector, online computation ***")
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

    b = assemble(v_0[0]*dx + v_0[1]*dx + v_1*dx)
    b_N = assemble(v_N_0[0]*dx + v_N_0[1]*dx + v_N_1*dx)

    b_dofs = evaluate_sparse_vector_at_dofs(b, dofs)
    b_N_reduced_dofs = evaluate_sparse_vector_at_dofs(b_N, reduced_dofs)
    
    log(PROGRESS, "b at dofs:\n" + str(b_dofs))
    log(PROGRESS, "b_N at reduced dofs:\n" + str(b_N_reduced_dofs))
    log(PROGRESS, "Error:\n" + str(b_dofs - b_N_reduced_dofs))
    
    assert isclose(b_dofs, b_N_reduced_dofs)

# ~~~ Collapsed case ~~~ #
def CollapsedFunctionSpaces(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    U = FunctionSpace(mesh, element)
    V = U.sub(0).collapse()
    return (V, U)

# === Matrix computation === #
@generate_meshes
def test_reduced_mesh_io_collapsed_matrix(mesh, tempdir):
    test_reduced_mesh_save_collapsed_matrix(mesh, tempdir)
    test_reduced_mesh_load_collapsed_matrix(mesh, tempdir)

@generate_meshes
def test_reduced_mesh_save_collapsed_matrix(mesh, save_tempdir):
    log(PROGRESS, "*** Collapsed case, matrix, offline computation ***")
    (V, U) = CollapsedFunctionSpaces(mesh)
    reduced_mesh = ReducedMesh((V, U))
    dofs = [(2, 1), (48, 33), (40, 12), (31, 39)]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_collapsed_matrix")
    
    _test_reduced_mesh_collapsed_matrix(V, U, reduced_mesh)
    
@generate_meshes
def test_reduced_mesh_load_collapsed_matrix(mesh, load_tempdir):
    log(PROGRESS, "*** Collapsed case, matrix, online computation ***")
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
    
    A = assemble(inner(u_0, v)*dx + u_1*v[0]*dx)
    A_N = assemble(inner(u_N_0, v_N)*dx + u_N_1*v_N[0]*dx)

    A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
    A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

    log(PROGRESS, "A at dofs:\n" + str(A_dofs))
    log(PROGRESS, "A_N at reduced dofs:\n" + str(A_N_reduced_dofs))
    log(PROGRESS, "Error:\n" + str(A_dofs - A_N_reduced_dofs))
    
    assert isclose(A_dofs, A_N_reduced_dofs)
    
# === Vector computation === #
@generate_meshes
def test_reduced_mesh_io_collapsed_vector(mesh, tempdir):
    test_reduced_mesh_save_collapsed_vector(mesh, tempdir)
    test_reduced_mesh_load_collapsed_vector(mesh, tempdir)
    
@generate_meshes
def test_reduced_mesh_save_collapsed_vector(mesh, save_tempdir):
    log(PROGRESS, "*** Collapsed case, vector, offline computation ***")
    (V, _) = CollapsedFunctionSpaces(mesh)
    reduced_mesh = ReducedMesh((V, ))
    dofs = [(2, ), (48, ), (40, ), (11, )]

    for pair in dofs:
        log(PROGRESS, "Adding " + str(pair))
        reduced_mesh.append(pair)
        
        reduced_mesh.save(save_tempdir, "test_reduced_mesh_collapsed_vector")
        
    _test_reduced_mesh_collapsed_vector(V, reduced_mesh)

@generate_meshes
def test_reduced_mesh_load_collapsed_vector(mesh, load_tempdir):
    log(PROGRESS, "*** Collapsed case, vector, online computation ***")
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

    b = assemble(v[0]*dx + v[1]*dx)
    b_N = assemble(v_N[0]*dx + v_N[1]*dx)

    b_dofs = evaluate_sparse_vector_at_dofs(b, dofs)
    b_N_reduced_dofs = evaluate_sparse_vector_at_dofs(b_N, reduced_dofs)
    
    log(PROGRESS, "b at dofs:\n" + str(b_dofs))
    log(PROGRESS, "b_N at reduced dofs:\n" + str(b_N_reduced_dofs))
    log(PROGRESS, "Error:\n" + str(b_dofs - b_N_reduced_dofs))
