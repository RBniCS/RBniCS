# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pickle
import pytest
from logging import DEBUG, getLogger
from numpy import allclose, ndarray as array
import matplotlib
import matplotlib.pyplot as plt
from dolfin import (cells, facets, FiniteElement, FunctionSpace, HDF5File, MeshFunction, MixedElement, MPI,
                    UnitSquareMesh, VectorElement, vertices)
from dolfin_utils.test import fixture as module_fixture
try:
    from fenicstools import DofMapPlotter as FEniCSToolsDofMapPlotter
except ImportError:
    has_fenicstools = False
else:
    has_fenicstools = True
from rbnics.backends.dolfin.wrapping import (convert_functionspace_to_submesh, convert_meshfunctions_to_submesh,
                                             create_submesh, map_functionspaces_between_mesh_and_submesh)
from rbnics.backends.dolfin.wrapping.create_submesh import logger as create_submesh_logger
from rbnics.utils.test import enable_logging

# Data directory
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_create_submesh")

# Logger
test_logger = getLogger("tests/unit/test_create_submesh.py")
enable_create_submesh_logging = enable_logging({create_submesh_logger: DEBUG, test_logger: DEBUG})


# Mesh
@module_fixture
def mesh():
    mesh = UnitSquareMesh(3, 3)
    assert MPI.size(mesh.mpi_comm()) in (1, 2, 3, 4)
    # 1 processor        -> test serial case
    # 2 and 3 processors -> test case where submesh in contained only on one processor
    # 4 processors       -> test case where submesh is shared by two processors,
    #                       resulting in shared facets and vertices
    return mesh


# Mesh subdomains
@module_fixture
def subdomains(mesh):
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in cells(mesh):
        subdomains.array()[c.index()] = c.global_index()
    return subdomains


# Mesh boundaries
@module_fixture
def boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    for f in facets(mesh):
        boundaries.array()[f.index()] = 0
        for v in vertices(f):
            boundaries.array()[f.index()] += v.global_index()
    return boundaries


# Submesh markers
@module_fixture
def submesh_markers(mesh):
    markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
    hdf = HDF5File(mesh.mpi_comm(), os.path.join(data_dir, "markers.h5"), "r")
    hdf.read(markers, "/cells")
    return markers


# Submesh
@module_fixture
def submesh(mesh, submesh_markers):
    return create_submesh(mesh, submesh_markers)


# Internal: submesh subdomains and boundaries
@module_fixture
def _submesh_subdomains_boundaries(mesh, submesh, subdomains, boundaries):
    return convert_meshfunctions_to_submesh(mesh, submesh, [subdomains, boundaries])


# Submesh subdomains
@module_fixture
def submesh_subdomains(_submesh_subdomains_boundaries):
    return _submesh_subdomains_boundaries[0]


# Submesh boundaries
@module_fixture
def submesh_boundaries(_submesh_subdomains_boundaries):
    return _submesh_subdomains_boundaries[1]


# Auxiliary functions for array asserts
def array_save(arr, directory, filename):
    assert isinstance(arr, array)
    with open(os.path.join(directory, filename), "wb") as outfile:
        pickle.dump(arr, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def array_assert_equal(arr, directory, filename):
    assert isinstance(arr, array)
    with open(os.path.join(directory, filename), "rb") as infile:
        arr_in = pickle.load(infile)
    assert (arr == arr_in).all()


# Auxiliary functions for dict asserts
def dict_save(dic, directory, filename):
    assert isinstance(dic, dict)
    with open(os.path.join(directory, filename), "wb") as outfile:
        pickle.dump(dic, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def dict_assert_equal(dic, directory, filename):
    assert isinstance(dic, dict)
    with open(os.path.join(directory, filename), "rb") as infile:
        dic_in = pickle.load(infile)
    assert dic.keys() == dic_in.keys()
    for key in dic.keys():
        dic_value = dic[key]
        dic_in_value = dic_in[key]
        if isinstance(dic_value, set) and isinstance(dic_in_value, array):
            # pybind11 has changed the return type of shared entities
            assert dic_value == set(dic_in_value.tolist())
        else:
            assert dic_value == dic_in_value


# Auxiliary functions for list asserts
def list_save(lis, directory, filename):
    assert isinstance(lis, list)
    with open(os.path.join(directory, filename), "wb") as outfile:
        pickle.dump(lis, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def list_assert_equal(lis, directory, filename):
    assert isinstance(lis, list)
    with open(os.path.join(directory, filename), "rb") as infile:
        lis_in = pickle.load(infile)
    assert lis == lis_in


# Auxiliary functions to open DofMapPlotter
def assert_mesh_plotter(mesh, submesh, key, directory, filename):
    V = FunctionSpace(mesh, "Real", 0)
    submesh_V = FunctionSpace(submesh, "Real", 0)
    assert_dof_map_plotter(V, submesh_V, key, directory, filename)


def assert_dof_map_plotter(V, submesh_V, key, directory, filename):
    dmp_V = DofMapPlotter(V, key)
    dmp_submesh_V = DofMapPlotter(submesh_V, key)
    plt.show()
    dmp_V.save(directory, filename + "__mesh_plot_")
    dmp_submesh_V.save(directory, filename + "__submesh_plot_")
    dmp_V.assert_equal(data_dir, filename + "__mesh_plot_")
    dmp_submesh_V.assert_equal(data_dir, filename + "__submesh_plot_")


# Auxiliary functions for function space definition
def EllipticFunctionSpace(mesh):
    return FunctionSpace(mesh, "CG", 2)


def StokesFunctionSpace(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return FunctionSpace(mesh, element)


# DofMapPlotter
if has_fenicstools:
    # Patch matplotlib.axes.Axes.text to store the text before writing it
    original_text = matplotlib.axes.Axes.text

    def custom_text(self, x, y, s, *args, **kwargs):
        original_text(self, x, y, s, *args, **kwargs)
        if not hasattr(self, "_text_storage"):
            self._text_storage = dict()
        self._text_storage[s] = (x, y)

    matplotlib.axes.Axes.text = custom_text

    # Event that mimics a keypress
    class Event(object):
        def __init__(self, key):
            self.key = key

    # Custom dof map plotter
    class DofMapPlotter(FEniCSToolsDofMapPlotter):
        def __init__(self, V, key):
            FEniCSToolsDofMapPlotter.__init__(self, V)
            self.plot()
            assert len(self.plots) == 1
            assert key in ("C", "T", "V", "D")
            if key in ("C", "T", "V"):
                self.plots[0].mesh_entity_handler.__call__(Event(key))
            elif key in ("D", ):
                self.plots[0].dof_handler.__call__(Event(key))
            self.ax = self.plots[0].mesh_entity_handler.axes

        def save(self, directory, filename):
            assert hasattr(self.ax, "_text_storage")
            with open(os.path.join(directory, filename + "_size_" + str(self.mpi_size)
                      + "_rank_" + str(self.mpi_rank) + ".pkl"), "wb") as outfile:
                pickle.dump(self.ax._text_storage, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        def assert_equal(self, directory, filename):
            with open(os.path.join(directory, filename + "_size_" + str(self.mpi_size)
                      + "_rank_" + str(self.mpi_rank) + ".pkl"), "rb") as infile:
                text_storage = pickle.load(infile)
            assert hasattr(self.ax, "_text_storage")
            assert text_storage.keys() == self.ax._text_storage.keys()
            for key in text_storage:
                assert allclose(text_storage[key], self.ax._text_storage[key])
else:
    class DofMapPlotter(object):
        def __init__(self, V, key):
            pass

        def save(self, directory, filename):
            pass

        def assert_equal(self, directory, filename):
            pass

# === NOTE: most of the following tests require interactivity because === #
# ===       cells may be moved from the owning process to another     === #
# ===       process which otherwise would have no cell.               === #
# ===       The same communication should be done in these tests to   === #
# ===       assert the results. Instead, we rely on fenicstools'      === #
# ===       DofMapPlotter and compare its internal state.             === #


# Test mesh to submesh global cell indices
@enable_create_submesh_logging
def test_mesh_to_submesh_global_cell_indices(mesh, submesh, tempdir):
    test_logger.log(DEBUG, "Mesh to submesh global cell indices:")
    for (mesh_local_index, submesh_local_index) in submesh.mesh_to_submesh_cell_local_indices.items():
        mesh_global_index = mesh.topology().global_indices(mesh.topology().dim())[mesh_local_index]
        submesh_global_index = submesh.topology().global_indices(submesh.topology().dim())[submesh_local_index]
        test_logger.log(DEBUG, "\t" + str(mesh_global_index) + " -> " + str(submesh_global_index))
    assert_mesh_plotter(mesh, submesh, "C", tempdir, "test_mesh_to_submesh_global_cell_indices")
    filename = ("test_mesh_to_submesh_global_cell_indices" + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    dict_save(submesh.mesh_to_submesh_cell_local_indices, tempdir, filename)
    dict_assert_equal(submesh.mesh_to_submesh_cell_local_indices, data_dir, filename)


# Test submesh to mesh global cell indices
@enable_create_submesh_logging
def test_submesh_to_mesh_global_cell_indices(mesh, submesh, tempdir):
    test_logger.log(DEBUG, "Submesh to mesh global cell indices:")
    for (submesh_local_index, mesh_local_index) in enumerate(submesh.submesh_to_mesh_cell_local_indices):
        submesh_global_index = submesh.topology().global_indices(submesh.topology().dim())[submesh_local_index]
        mesh_global_index = mesh.topology().global_indices(mesh.topology().dim())[mesh_local_index]
        test_logger.log(DEBUG, "\t" + str(submesh_global_index) + " -> " + str(mesh_global_index))
    assert_mesh_plotter(mesh, submesh, "C", tempdir, "test_submesh_to_mesh_global_cell_indices")
    filename = ("test_submesh_to_mesh_global_cell_indices" + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    array_save(submesh.submesh_to_mesh_cell_local_indices, tempdir, filename)
    array_assert_equal(submesh.submesh_to_mesh_cell_local_indices, data_dir, filename)


# Test that the cell numbering is independent on the number of processors, and that
# fake cells have the largest numbering
@enable_create_submesh_logging
def test_submesh_global_cell_numbering_independent_on_mpi(mesh, submesh_markers, submesh, tempdir):
    cell_markers = dict()
    cell_centroids = dict()
    for submesh_cell in cells(submesh):
        submesh_local_index = submesh_cell.index()
        submesh_global_index = submesh.topology().global_indices(submesh.topology().dim())[submesh_local_index]
        mesh_local_index = submesh.submesh_to_mesh_cell_local_indices[submesh_local_index]
        cell_markers[submesh_global_index] = submesh_markers.array()[mesh_local_index]
        cell_centroids[submesh_global_index] = [submesh_cell.midpoint()[i] for i in range(submesh.topology().dim())]
    output_filename = ("test_submesh_cell_numbering_independent_on_mpi__size_" + str(MPI.size(submesh.mpi_comm()))
                       + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    with open(os.path.join(tempdir, output_filename), "wb") as outfile:
        pickle.dump(cell_centroids, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    input_filename = "test_submesh_cell_numbering_independent_on_mpi__size_1_rank_0.pkl"
    with open(os.path.join(data_dir, input_filename), "rb") as infile:
        serial_cell_centroids = pickle.load(infile)
    for submesh_global_index in cell_centroids.keys():
        if submesh_global_index < len(serial_cell_centroids):
            assert allclose(cell_centroids[submesh_global_index], serial_cell_centroids[submesh_global_index])
            assert cell_markers[submesh_global_index]
        else:
            assert not cell_markers[submesh_global_index]


# Test mesh to submesh global facet indices
@enable_create_submesh_logging
def test_mesh_to_submesh_global_facet_indices(mesh, submesh, tempdir):
    test_logger.log(DEBUG, "Mesh to submesh global facet indices:")
    for (mesh_local_index, submesh_local_index) in submesh.mesh_to_submesh_facet_local_indices.items():
        mesh_global_index = mesh.topology().global_indices(mesh.topology().dim() - 1)[mesh_local_index]
        submesh_global_index = submesh.topology().global_indices(submesh.topology().dim() - 1)[submesh_local_index]
        test_logger.log(DEBUG, "\t" + str(mesh_global_index) + " -> " + str(submesh_global_index))
    assert_mesh_plotter(mesh, submesh, "T", tempdir, "test_mesh_to_submesh_global_facet_indices")
    filename = ("test_mesh_to_submesh_global_facet_indices" + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    dict_save(submesh.mesh_to_submesh_facet_local_indices, tempdir, filename)
    dict_assert_equal(submesh.mesh_to_submesh_facet_local_indices, data_dir, filename)


# Test submesh to mesh global facet indices
@enable_create_submesh_logging
def test_submesh_to_mesh_global_facet_indices(mesh, submesh, tempdir):
    test_logger.log(DEBUG, "Submesh to mesh global facet indices:")
    for (submesh_local_index, mesh_local_index) in enumerate(submesh.submesh_to_mesh_facet_local_indices):
        submesh_global_index = submesh.topology().global_indices(submesh.topology().dim() - 1)[submesh_local_index]
        mesh_global_index = mesh.topology().global_indices(mesh.topology().dim() - 1)[mesh_local_index]
        test_logger.log(DEBUG, "\t" + str(submesh_global_index) + " -> " + str(mesh_global_index))
    assert_mesh_plotter(mesh, submesh, "T", tempdir, "test_submesh_to_mesh_global_facet_indices")
    filename = ("test_submesh_to_mesh_global_facet_indices" + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    list_save(submesh.submesh_to_mesh_facet_local_indices, tempdir, filename)
    list_assert_equal(submesh.submesh_to_mesh_facet_local_indices, data_dir, filename)


# Test mesh to submesh global vertex indices
@enable_create_submesh_logging
def test_mesh_to_submesh_global_vertex_indices(mesh, submesh, tempdir):
    test_logger.log(DEBUG, "Mesh to submesh global vertex indices:")
    for (mesh_local_index, submesh_local_index) in submesh.mesh_to_submesh_vertex_local_indices.items():
        mesh_global_index = mesh.topology().global_indices(0)[mesh_local_index]
        submesh_global_index = submesh.topology().global_indices(0)[submesh_local_index]
        test_logger.log(DEBUG, "\t" + str(mesh_global_index) + " -> " + str(submesh_global_index))
    assert_mesh_plotter(mesh, submesh, "V", tempdir, "test_mesh_to_submesh_global_vertex_indices")
    filename = ("test_mesh_to_submesh_global_vertex_indices" + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    dict_save(submesh.mesh_to_submesh_vertex_local_indices, tempdir, filename)
    dict_assert_equal(submesh.mesh_to_submesh_vertex_local_indices, data_dir, filename)


# Test submesh to mesh global vertex indices
@enable_create_submesh_logging
def test_submesh_to_mesh_global_vertex_indices(mesh, submesh, tempdir):
    test_logger.log(DEBUG, "Submesh to mesh global vertex indices:")
    for (submesh_local_index, mesh_local_index) in enumerate(submesh.submesh_to_mesh_vertex_local_indices):
        submesh_global_index = submesh.topology().global_indices(0)[submesh_local_index]
        mesh_global_index = mesh.topology().global_indices(0)[mesh_local_index]
        test_logger.log(DEBUG, "\t" + str(submesh_global_index) + " -> " + str(mesh_global_index))
    assert_mesh_plotter(mesh, submesh, "V", tempdir, "test_submesh_to_mesh_global_vertex_indices")
    filename = ("test_submesh_to_mesh_global_vertex_indices" + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    array_save(submesh.submesh_to_mesh_vertex_local_indices, tempdir, filename)
    array_assert_equal(submesh.submesh_to_mesh_vertex_local_indices, data_dir, filename)


# Test shared entities detection
@enable_create_submesh_logging
def test_shared_entities_detection(mesh, submesh, tempdir):
    dim_to_text = {
        submesh.topology().dim(): "cells",
        submesh.topology().dim() - 1: "facets",
        0: "vertices"
    }
    for dim in [submesh.topology().dim(), submesh.topology().dim() - 1, 0]:
        shared_entities = submesh.topology().shared_entities(dim)
        test_logger.log(DEBUG, "Submesh shared indices for " + str(dim_to_text[dim]))
        test_logger.log(DEBUG, str(shared_entities))
        filename = ("test_shared_entities_detection__dim_" + str(dim) + "__size_" + str(MPI.size(submesh.mpi_comm()))
                    + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
        dict_save(shared_entities, tempdir, filename)
        dict_assert_equal(shared_entities, data_dir, filename)


# Test mesh to submesh dof map
@enable_create_submesh_logging
@pytest.mark.parametrize("FunctionSpace", (EllipticFunctionSpace, StokesFunctionSpace))
def test_mesh_to_submesh_dof_map(mesh, FunctionSpace, submesh_markers, submesh, tempdir):
    test_logger.log(DEBUG, "Mesh to submesh dofs")
    V = FunctionSpace(mesh)
    submesh_V = convert_functionspace_to_submesh(V, submesh)
    (mesh_dofs_to_submesh_dofs, submesh_dofs_to_mesh_dofs) = map_functionspaces_between_mesh_and_submesh(
        V, mesh, submesh_V, submesh)
    test_logger.log(DEBUG, "Local mesh dofs ownership range: " + str(V.dofmap().ownership_range()))
    for (mesh_dof, submesh_dof) in mesh_dofs_to_submesh_dofs.items():
        test_logger.log(DEBUG, "\t" + str(mesh_dof) + " -> " + str(submesh_dof))
    assert_dof_map_plotter(V, submesh_V, "D", tempdir, "test_mesh_to_submesh_dof_map_" + FunctionSpace.__name__)
    filename = ("test_mesh_to_submesh_dof_map_" + FunctionSpace.__name__ + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    dict_save(mesh_dofs_to_submesh_dofs, tempdir, filename)
    dict_assert_equal(mesh_dofs_to_submesh_dofs, data_dir, filename)


# Test submesh to mesh dof map
@enable_create_submesh_logging
@pytest.mark.parametrize("FunctionSpace", (EllipticFunctionSpace, StokesFunctionSpace))
def test_submesh_to_mesh_dof_map(mesh, FunctionSpace, submesh_markers, submesh, tempdir):
    test_logger.log(DEBUG, "Submesh to mesh dofs")
    V = FunctionSpace(mesh)
    submesh_V = convert_functionspace_to_submesh(V, submesh)
    (mesh_dofs_to_submesh_dofs, submesh_dofs_to_mesh_dofs) = map_functionspaces_between_mesh_and_submesh(
        V, mesh, submesh_V, submesh)
    test_logger.log(DEBUG, "Local submesh dofs ownership range: " + str(submesh_V.dofmap().ownership_range()))
    for (submesh_dof, mesh_dof) in submesh_dofs_to_mesh_dofs.items():
        test_logger.log(DEBUG, "\t" + str(submesh_dof) + " -> " + str(mesh_dof))
    assert_dof_map_plotter(V, submesh_V, "D", tempdir, "test_submesh_to_mesh_dof_map_" + FunctionSpace.__name__)
    filename = ("test_submesh_to_mesh_dof_map_" + FunctionSpace.__name__ + "_size_" + str(MPI.size(submesh.mpi_comm()))
                + "_rank_" + str(MPI.rank(submesh.mpi_comm())) + ".pkl")
    dict_save(submesh_dofs_to_mesh_dofs, tempdir, filename)
    dict_assert_equal(submesh_dofs_to_mesh_dofs, data_dir, filename)
