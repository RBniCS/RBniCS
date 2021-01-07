# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from logging import DEBUG, getLogger
from mpi4py.MPI import MAX
from dolfin import cells, has_hdf5, has_hdf5_parallel, Mesh, MeshFunction
from rbnics.backends.abstract import ReducedMesh as AbstractReducedMesh
from rbnics.backends.dolfin.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.dolfin.wrapping import (build_dof_map_reader_mapping, build_dof_map_writer_mapping,
                                             create_submesh, convert_meshfunctions_to_submesh,
                                             convert_functionspace_to_submesh,
                                             evaluate_basis_functions_matrix_at_dofs,
                                             evaluate_sparse_function_at_dofs, FunctionSpace,
                                             map_functionspaces_between_mesh_and_submesh)
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import _sub_from_tuple
from rbnics.backends.dolfin.wrapping.get_auxiliary_problem_for_non_parametrized_function import (
    AuxiliaryProblemForNonParametrizedFunction)
from rbnics.utils.decorators import (abstractmethod, BackendFor, get_reduced_problem_from_problem,
                                     get_reduction_method_from_problem, is_training_finished, ModuleWrapper)
from rbnics.utils.io import ExportableList, Folders
from rbnics.utils.mpi import parallel_io
from rbnics.utils.test import PatchInstanceMethod

logger = getLogger("rbnics/backends/dolfin/reduced_mesh.py")

if not has_hdf5() or not has_hdf5_parallel():
    from dolfin import File as ASCIIFile

    class XMLFile(object):
        def __init__(self, mpi_comm, mesh_dim, filename, rw_mode):
            assert mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
            self.filename = filename + ".xml"

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            pass

    class MeshFile(XMLFile):
        def read(self):
            return Mesh(self.filename)

        def write(self, mesh):
            ASCIIFile(self.filename) << mesh

    class MeshFunctionFile(XMLFile):
        def read(self, value_type, mesh, subdomain_dim):
            return MeshFunction(value_type, mesh, self.filename)

        def write(self, mesh_function):
            ASCIIFile(self.filename) << mesh_function
else:
    from dolfin import HDF5File, XDMFFile

    class BinaryFile(object):
        def __init__(self, mpi_comm, mesh_dim, filename, rw_mode):
            self.h5_file = HDF5File(mpi_comm, filename + "_checkpoint.h5", rw_mode)
            self.xdmf_file = XDMFFile(mpi_comm, filename + ".xdmf")

        def __enter__(self):
            self.h5_file.__enter__()
            self.xdmf_file.__enter__()
            return self

        def __exit__(self, *exc_info):
            self.h5_file.__exit__(*exc_info)
            self.xdmf_file.__exit__(*exc_info)

    class MeshFile(BinaryFile):
        def read(self):
            reduced_mesh = Mesh()
            self.h5_file.read(reduced_mesh, "/mesh", False)
            return reduced_mesh

        def write(self, mesh):
            self.h5_file.write(mesh, "/mesh")
            self.xdmf_file.write(mesh)

    class MeshFunctionFile(BinaryFile):
        def read(self, value_type, mesh, subdomain_dim):
            reduced_subdomain = MeshFunction(value_type, mesh, subdomain_dim)
            self.h5_file.read(reduced_subdomain, "/mesh_function")
            return reduced_subdomain

        def write(self, mesh_function):
            self.h5_file.write(mesh_function, "/mesh_function")
            self.xdmf_file.write(mesh_function)


def BasicReducedMesh(backend, wrapping):

    class _BasicReducedMesh(AbstractReducedMesh):
        def __init__(self, V, subdomain_data=None, auxiliary_problems_and_components=None, **kwargs):
            AbstractReducedMesh.__init__(self, V)
            #
            assert isinstance(V, tuple)
            assert len(V) in (1, 2)
            if len(V) == 2:
                assert V[0].mesh().ufl_domain() == V[1].mesh().ufl_domain()
            self.mesh = V[0].mesh()
            self.mpi_comm = self.mesh.mpi_comm()
            self.V = V
            self.subdomain_data = subdomain_data
            self.auxiliary_problems_and_components = auxiliary_problems_and_components

            # Detect if **kwargs are provided by the copy constructor in __getitem__
            if "copy_from" in kwargs:
                copy_from = kwargs["copy_from"]
                assert "key_as_slice" in kwargs
                key_as_slice = kwargs["key_as_slice"]
                assert "key_as_int" in kwargs
                key_as_int = kwargs["key_as_int"]
            else:
                copy_from = None
                key_as_slice = None
                key_as_int = None

            # Prepare storage for an helper dof to cell dict
            self.dof_to_cells = tuple()  # of size len(V)
            # ... which is not initialized in the constructor to avoid wasting time online
            # ... since it is only needed offline in the append() method

            # Cell functions to mark cells (on the full mesh)
            self.reduced_mesh_markers = dict()  # from N to MeshFunction
            # ... which again is not initialized here for performance reasons

            # DOFs list (of the full mesh) that need to be added at each N
            self.reduced_mesh_dofs_list = list()  # list (of size N) of tuple (of size len(V)) of dofs
            if copy_from is not None:
                self.reduced_mesh_dofs_list.extend(copy_from.reduced_mesh_dofs_list[key_as_slice])
            # Prepare storage for helper mapping needed for I/O
            self.reduced_mesh_dofs_list__dof_map_writer_mapping = tuple()  # of size len(V)
            self.reduced_mesh_dofs_list__dof_map_reader_mapping = tuple()  # of size len(V)
            # ... which will be initialized as needed in the save and load methods

            # Reduced meshes, for all N
            self.reduced_mesh = dict()  # from N to Mesh
            if copy_from is not None:
                self.reduced_mesh[key_as_int] = copy_from.reduced_mesh[key_as_int]

            # Reduced subdomain data, for all N
            self.reduced_subdomain_data = dict()  # from N to dict from mesh MeshFunction to reduced_mesh MeshFunction
            if copy_from is not None:
                self.reduced_subdomain_data[key_as_int] = copy_from.reduced_subdomain_data[key_as_int]

            # Reduced function spaces, for all N
            self.reduced_function_spaces = dict()  # from N to tuple (of size len(V)) of FunctionSpace
            if copy_from is not None:
                self.reduced_function_spaces[key_as_int] = copy_from.reduced_function_spaces[key_as_int]

            # DOFs list (of the reduced mesh) that need to be added at each N
            self.reduced_mesh_reduced_dofs_list = dict()  # from N to list of tuple (of size len(V)) of dofs
            if copy_from is not None:
                self.reduced_mesh_reduced_dofs_list[key_as_int] = copy_from.reduced_mesh_reduced_dofs_list[key_as_int]
            # Prepare storage for helper mapping needed for I/O
            self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping = dict()  # from N to tuple (of size len(V))
            self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping = dict()  # from N to tuple (of size len(V))
            # ... which will be initialized as needed in the save and load methods

            # == The following members are related to auxiliary basis functions for nonlinear terms. == #
            # Spaces for auxiliary basis functions
            self._auxiliary_reduced_function_space = dict()  # from (problem, component) to dict from N to FunctionSpace
            if copy_from is not None:
                self._auxiliary_reduced_function_space = copy_from._auxiliary_reduced_function_space
            else:
                if auxiliary_problems_and_components is not None:
                    for key in auxiliary_problems_and_components:
                        self._auxiliary_reduced_function_space[key] = dict()
            # Mapping between DOFs on the reduced mesh and DOFs on the full mesh for auxiliary basis functions
            # from (problem, component) to dict from N to dict from int to int
            self._auxiliary_dofs_to_reduced_dofs = dict()
            if copy_from is not None:
                self._auxiliary_dofs_to_reduced_dofs = copy_from._auxiliary_dofs_to_reduced_dofs
            else:
                if auxiliary_problems_and_components is not None:
                    for key in auxiliary_problems_and_components:
                        self._auxiliary_dofs_to_reduced_dofs[key] = dict()
            # Auxiliary basis functions, as dict from (problem, component) to dict from N to BasisFunctionsMatrix
            self._auxiliary_basis_functions_matrix = dict()
            self._auxiliary_basis_functions_matrix_save_patched = dict()  # from problem to bool
            self._auxiliary_basis_functions_matrix_load_patched = dict()  # from problem to bool
            if copy_from is not None:
                self._auxiliary_basis_functions_matrix = copy_from._auxiliary_basis_functions_matrix
                self._auxiliary_basis_functions_matrix_save_patched = (
                    copy_from._auxiliary_basis_functions_matrix_save_patched)
                self._auxiliary_basis_functions_matrix_load_patched = (
                    copy_from._auxiliary_basis_functions_matrix_load_patched)
            else:
                if auxiliary_problems_and_components is not None:
                    for key in auxiliary_problems_and_components:
                        self._auxiliary_basis_functions_matrix[key] = dict()
                        self._auxiliary_basis_functions_matrix_save_patched[key] = False
                        self._auxiliary_basis_functions_matrix_load_patched[key] = False
            # Auxiliary function interpolator
            self._auxiliary_function_interpolator = dict()  # from (problem, component) to dict from N to function
            if copy_from is not None:
                self._auxiliary_function_interpolator = copy_from._auxiliary_function_interpolator
            else:
                if auxiliary_problems_and_components is not None:
                    for key in auxiliary_problems_and_components:
                        self._auxiliary_function_interpolator[key] = dict()
            # Prepare storage for helper mapping needed for I/O
            self._auxiliary_dofs__dof_map_writer_mapping = dict()  # from problem
            self._auxiliary_dofs__dof_map_reader_mapping = dict()  # from problem
            self._auxiliary_reduced_dofs__dof_map_writer_mapping = dict()  # from (problem, component) to dict from N
            self._auxiliary_reduced_dofs__dof_map_reader_mapping = dict()  # from (problem, component) to dict from N
            # ... which will be initialized as needed in the save and load methods

        def append(self, global_dofs):
            self._init_for_append_if_needed()
            # Consistency checks
            assert isinstance(global_dofs, tuple)
            assert len(global_dofs) == len(self.V)
            self.reduced_mesh_dofs_list.append(global_dofs)
            # Mark all cells
            N = self._get_next_index()
            reduced_mesh_markers = self.reduced_mesh_markers[N]
            for (component, global_dof) in enumerate(global_dofs):
                global_dof_found = 0
                if global_dof in self.dof_to_cells[component]:
                    global_dof_found = 1
                    for cell in self.dof_to_cells[component][global_dof]:
                        reduced_mesh_markers[cell] = True
                global_dof_found = self.mpi_comm.allreduce(global_dof_found, op=MAX)
                assert global_dof_found == 1
            # Actually update to data structures using updated cells marker
            self._update()

        def _update(self):
            N = self._get_next_index()
            # Create submesh
            reduced_mesh = wrapping.create_submesh(self.mesh, self.reduced_mesh_markers[N])
            self.reduced_mesh[N] = reduced_mesh
            # Create subdomain data on submesh
            if self.subdomain_data is not None:
                reduced_subdomain_data_list = wrapping.convert_meshfunctions_to_submesh(
                    self.mesh, reduced_mesh, self.subdomain_data)
                reduced_subdomain_data = dict()
                assert len(self.subdomain_data) == len(reduced_subdomain_data_list)
                for (subdomain, reduced_subdomain) in zip(self.subdomain_data, reduced_subdomain_data_list):
                    reduced_subdomain_data[subdomain] = reduced_subdomain
                self.reduced_subdomain_data[N] = reduced_subdomain_data
            else:
                self.reduced_subdomain_data[N] = None
            # Store the FunctionSpace V on the reduced mesh, as well as the map between DOFs on V and reduced_V
            reduced_function_spaces = list()
            dofs__to__reduced_dofs = list()  # of size len(V)
            for (component, V_component) in enumerate(self.V):
                reduced_function_space_component = wrapping.convert_functionspace_to_submesh(
                    V_component, reduced_mesh, self._get_reduced_function_space_type(V_component))
                reduced_function_spaces.append(reduced_function_space_component)
                (dofs__to__reduced_dofs_component, _) = wrapping.map_functionspaces_between_mesh_and_submesh(
                    V_component, self.mesh, reduced_function_space_component, reduced_mesh)
                dofs__to__reduced_dofs.append(dofs__to__reduced_dofs_component)
                logger.log(DEBUG, "DOFs to reduced DOFs (component " + str(component) + ") is "
                           + str(dofs__to__reduced_dofs[component]))
            self.reduced_function_spaces[N] = tuple(reduced_function_spaces)
            # ... and fill in reduced_mesh_reduced_dofs_list ...
            reduced_mesh_reduced_dofs_list = list()
            for dofs in self.reduced_mesh_dofs_list:
                reduced_dofs = list()
                for (component, dof) in enumerate(dofs):
                    dof_processor = -1
                    reduced_dof = None
                    if dof in dofs__to__reduced_dofs[component]:
                        reduced_dof = dofs__to__reduced_dofs[component][dof]
                        dof_processor = self.mpi_comm.rank
                    dof_processor = self.mpi_comm.allreduce(dof_processor, op=MAX)
                    assert dof_processor >= 0
                    reduced_dofs.append(self.mpi_comm.bcast(reduced_dof, root=dof_processor))
                assert len(reduced_dofs) in (1, 2)
                reduced_mesh_reduced_dofs_list.append(tuple(reduced_dofs))
            logger.log(DEBUG, "Reduced DOFs list " + str(reduced_mesh_reduced_dofs_list))
            logger.log(DEBUG, "corresponding to DOFs list " + str(self.reduced_mesh_dofs_list))
            self.reduced_mesh_reduced_dofs_list[N] = reduced_mesh_reduced_dofs_list
            # Finally, update terms related to auxiliary problems
            self._update_auxiliary()

        def _init_for_append_if_needed(self):
            # Initialize dof to cells map only the first time
            if len(self.dof_to_cells) == 0:
                self.dof_to_cells = list()  # of size len(V)
                for (component, V_component) in enumerate(self.V):
                    dof_to_cells = self._compute_dof_to_cells(V_component)
                    # Debugging
                    logger.log(DEBUG, "DOFs to cells map (component " + str(component) + ") on processor "
                               + str(self.mpi_comm.rank) + ":")
                    for (global_dof, cells_) in dof_to_cells.items():
                        logger.log(DEBUG, "\t" + str(global_dof) + ": " + str([cell.global_index()
                                                                              for cell in cells_]))
                    # Add to storage
                    self.dof_to_cells.append(dof_to_cells)
                self.dof_to_cells = tuple(self.dof_to_cells)
            # Initialize cells marker
            N = self._get_next_index()
            reduced_mesh_markers = MeshFunction("bool", self.mesh, self.mesh.geometry().dim())
            reduced_mesh_markers.set_all(False)
            if N > 0:
                reduced_mesh_markers.array()[:] = self.reduced_mesh_markers[N - 1].array()
            assert N not in self.reduced_mesh_markers
            self.reduced_mesh_markers[N] = reduced_mesh_markers

        def _update_auxiliary(self):
            if self.auxiliary_problems_and_components is not None:
                for key in self.auxiliary_problems_and_components:
                    (auxiliary_problem, component) = key
                    self._update_auxiliary_reduced_function_space(auxiliary_problem, component)
                    self._update_auxiliary_function_interpolator(auxiliary_problem, component)
                    if not isinstance(auxiliary_problem, wrapping.AuxiliaryProblemForNonParametrizedFunction):
                        if is_training_finished(auxiliary_problem):
                            self._update_auxiliary_basis_functions_matrix(auxiliary_problem, component)
                        else:
                            pass  # will be computed when training is finished (see _save_auxiliary)

        @abstractmethod
        def _compute_dof_to_cells(self, V_component):
            pass

        @staticmethod
        @abstractmethod
        def _get_reduced_function_space_type(V_component):
            pass

        def _update_auxiliary_reduced_function_space(self, auxiliary_problem, component, index=None):
            assert isinstance(component, tuple)
            assert len(component) > 0
            index = self._get_dict_index(index)
            auxiliary_V = _sub_from_tuple(auxiliary_problem.V, component)
            key = (auxiliary_problem, component)
            logger.log(DEBUG, "Updating auxiliary reduced function space for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            assert index not in self._auxiliary_reduced_function_space[key]
            assert index not in self._auxiliary_dofs_to_reduced_dofs[key]
            auxiliary_reduced_V = wrapping.convert_functionspace_to_submesh(
                auxiliary_V, self.reduced_mesh[index], self._get_auxiliary_reduced_function_space_type(auxiliary_V))
            self._auxiliary_reduced_function_space[key][index] = auxiliary_reduced_V
            # Get the map between DOFs on auxiliary_V and auxiliary_reduced_V
            (auxiliary_dofs_to_reduced_dofs, _) = wrapping.map_functionspaces_between_mesh_and_submesh(
                auxiliary_V, self.mesh, auxiliary_reduced_V, self.reduced_mesh[index])
            logger.log(DEBUG, "Auxiliary DOFs to reduced DOFs is " + str(auxiliary_dofs_to_reduced_dofs))
            self._auxiliary_dofs_to_reduced_dofs[key][index] = auxiliary_dofs_to_reduced_dofs

        @staticmethod
        @abstractmethod
        def _get_auxiliary_reduced_function_space_type(auxiliary_V):
            pass

        def _update_auxiliary_function_interpolator(self, auxiliary_problem, component, index=None):
            assert isinstance(component, tuple)
            assert len(component) > 0
            index = self._get_dict_index(index)
            key = (auxiliary_problem, component)
            logger.log(DEBUG, "Updating auxiliary function interpolator for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            assert index not in self._auxiliary_function_interpolator[key]
            auxiliary_reduced_V = self.get_auxiliary_reduced_function_space(auxiliary_problem, component, index)
            self._auxiliary_function_interpolator[key][index] = lambda fun: wrapping.evaluate_sparse_function_at_dofs(
                fun, self._auxiliary_dofs_to_reduced_dofs[key][index].keys(),
                auxiliary_reduced_V, self._auxiliary_dofs_to_reduced_dofs[key][index].values()
            )

        def _update_auxiliary_basis_functions_matrix(self, auxiliary_problem, component, index=None):
            assert isinstance(component, tuple)
            assert len(component) > 0
            index = self._get_dict_index(index)
            auxiliary_reduced_problem = get_reduced_problem_from_problem(auxiliary_problem)
            key = (auxiliary_problem, component)
            logger.log(DEBUG, "Updating auxiliary basis functions matrix for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            assert index not in self._auxiliary_basis_functions_matrix[key]
            auxiliary_reduced_V = self.get_auxiliary_reduced_function_space(auxiliary_problem, component, index)
            self._auxiliary_basis_functions_matrix[key][index] = self._init_auxiliary_basis_functions_matrix(
                auxiliary_reduced_problem, component, auxiliary_reduced_V)
            wrapping.evaluate_basis_functions_matrix_at_dofs(
                auxiliary_reduced_problem.basis_functions,
                self._auxiliary_dofs_to_reduced_dofs[key][index].keys(),
                self._auxiliary_basis_functions_matrix[key][index],
                self._auxiliary_dofs_to_reduced_dofs[key][index].values())

        def save(self, directory, filename):
            self._assert_dict_lengths()
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            full_directory.create()
            # Nmax
            self._save_Nmax(directory, filename)
            # reduced_mesh
            for (index, reduced_mesh) in self.reduced_mesh.items():
                mesh_filename = os.path.join(str(directory), filename, "reduced_mesh_" + str(index))
                with MeshFile(self.mesh.mpi_comm(), self.mesh.geometry().dim(), mesh_filename, "w") as output_file:
                    output_file.write(reduced_mesh)
            # cannot save reduced_function_spaces to file
            # reduced_subdomain_data
            if self.subdomain_data is not None:
                for (index, reduced_subdomain_data) in self.reduced_subdomain_data.items():
                    subdomain_index = 0
                    for (subdomain, reduced_subdomain) in reduced_subdomain_data.items():
                        subdomain_filename = os.path.join(
                            str(directory), filename,
                            "reduced_mesh_" + str(index) + "_subdomain_" + str(subdomain_index))
                        with MeshFunctionFile(self.mesh.mpi_comm(), self.mesh.geometry().dim(),
                                              subdomain_filename, "w") as output_file:
                            output_file.write(reduced_subdomain)
                        subdomain_index += 1
            # reduced_mesh_markers
            for (index, reduced_mesh_markers) in self.reduced_mesh_markers.items():
                marker_filename = os.path.join(str(directory), filename, "reduced_mesh_" + str(index) + "_markers")
                with MeshFunctionFile(self.mesh.mpi_comm(), self.mesh.geometry().dim(),
                                      marker_filename, "w") as output_file:
                    output_file.write(reduced_mesh_markers)
            # Init
            self._init_for_save_if_needed()
            # reduced_mesh_dofs_list
            exportable_reduced_mesh_dofs_list = ExportableList("pickle")
            for reduced_mesh_dof in self.reduced_mesh_dofs_list:
                for (component, reduced_mesh_dof__component) in enumerate(reduced_mesh_dof):
                    exportable_reduced_mesh_dofs_list.append(
                        self.reduced_mesh_dofs_list__dof_map_writer_mapping[component][reduced_mesh_dof__component])
            exportable_reduced_mesh_dofs_list.save(full_directory, "dofs")
            # reduced_mesh_reduced_dofs_list
            for (index, reduced_mesh_reduced_dofs_list) in self.reduced_mesh_reduced_dofs_list.items():
                exportable_reduced_mesh_reduced_dofs_list = ExportableList("pickle")
                for reduced_mesh_reduced_dof in reduced_mesh_reduced_dofs_list:
                    for (component, reduced_mesh_reduced_dof__component) in enumerate(reduced_mesh_reduced_dof):
                        exportable_reduced_mesh_reduced_dofs_list.append(
                            self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping[
                                index][component][reduced_mesh_reduced_dof__component])
                exportable_reduced_mesh_reduced_dofs_list.save(full_directory, "reduced_dofs_" + str(index))

            # Auxiliary problems
            self._save_auxiliary(directory, filename)

        def _save_Nmax(self, directory, filename):
            def save_Nmax_task():
                with open(os.path.join(str(directory), filename, "reduced_mesh.length"), "w") as length:
                    length.write(str(len(self.reduced_mesh)))
            parallel_io(save_Nmax_task, self.mpi_comm)

        def _init_for_save_if_needed(self):
            # Initialize dof map mappings for output
            if len(self.reduced_mesh_dofs_list__dof_map_writer_mapping) == 0:
                reduced_mesh_dofs_list__dof_map_writer_mapping = list()
                for V_component in self.V:
                    reduced_mesh_dofs_list__dof_map_writer_mapping.append(wrapping.build_dof_map_writer_mapping(
                        V_component))
                self.reduced_mesh_dofs_list__dof_map_writer_mapping = tuple(
                    reduced_mesh_dofs_list__dof_map_writer_mapping)

            # Initialize reduced dof mapping for output
            assert len(self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping) == len(self.reduced_mesh) - 1
            reduced_mesh_reduced_dofs_list__dof_map_writer_mapping = list()
            for reduced_V__component in self.reduced_function_spaces[len(self.reduced_mesh) - 1]:
                reduced_mesh_reduced_dofs_list__dof_map_writer_mapping.append(
                    wrapping.build_dof_map_writer_mapping(reduced_V__component))
            self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping[len(self.reduced_mesh) - 1] = tuple(
                reduced_mesh_reduced_dofs_list__dof_map_writer_mapping)

        def _save_auxiliary(self, directory, filename):
            if self.auxiliary_problems_and_components is not None:
                for key in self.auxiliary_problems_and_components:
                    (auxiliary_problem, component) = key
                    for index in range(len(self.reduced_mesh)):
                        self._save_auxiliary_reduced_function_space(
                            directory, filename, auxiliary_problem, component, index)
                        if not isinstance(auxiliary_problem, wrapping.AuxiliaryProblemForNonParametrizedFunction):
                            if is_training_finished(auxiliary_problem):
                                self._save_auxiliary_basis_functions_matrix(
                                    directory, filename, auxiliary_problem, component, index)
                            else:
                                self._patch_auxiliary_basis_functions_matrix_save(
                                    directory, filename, auxiliary_problem, component)

        def _save_auxiliary_reduced_function_space(self, directory, filename, auxiliary_problem, component, index):
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            full_directory.create()
            # Init
            self._init_for_auxiliary_save_if_needed(auxiliary_problem, component, index)
            # Save auxiliary dofs and reduced dofs
            logger.log(DEBUG, "Saving auxiliary reduced function space for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            key = (auxiliary_problem, component)
            auxiliary_dofs_to_reduced_dofs = self._auxiliary_dofs_to_reduced_dofs[key][index]
            # ... auxiliary dofs
            exportable_auxiliary_dofs = ExportableList("pickle")
            for auxiliary_dof in auxiliary_dofs_to_reduced_dofs.keys():
                exportable_auxiliary_dofs.append(self._auxiliary_dofs__dof_map_writer_mapping[
                    auxiliary_problem][auxiliary_dof])
            full_directory_plus_key_and_index__dofs = Folders.Folder(
                os.path.join(str(full_directory), "auxiliary_dofs", self._auxiliary_key_to_folder(key), str(index)))
            full_directory_plus_key_and_index__dofs.create()
            exportable_auxiliary_dofs.save(full_directory_plus_key_and_index__dofs, "auxiliary_dofs")
            # ... auxiliary reduced dofs
            exportable_auxiliary_reduced_dofs = ExportableList("pickle")
            for auxiliary_reduced_dof in auxiliary_dofs_to_reduced_dofs.values():
                exportable_auxiliary_reduced_dofs.append(
                    self._auxiliary_reduced_dofs__dof_map_writer_mapping[key][index][auxiliary_reduced_dof])
            full_directory_plus_key_and_index__reduced_dofs = Folders.Folder(
                os.path.join(
                    str(full_directory), "auxiliary_reduced_dofs", self._auxiliary_key_to_folder(key), str(index)))
            full_directory_plus_key_and_index__reduced_dofs.create()
            exportable_auxiliary_reduced_dofs.save(
                full_directory_plus_key_and_index__reduced_dofs, "auxiliary_reduced_dofs")

        def _init_for_auxiliary_save_if_needed(self, auxiliary_problem, component, index):
            # Initialize auxiliary dof map mappings and auxiliary reduced dof map mappings for output
            # ... auxiliary dof map mappings
            if auxiliary_problem not in self._auxiliary_dofs__dof_map_writer_mapping:
                self._auxiliary_dofs__dof_map_writer_mapping[
                    auxiliary_problem] = wrapping.build_dof_map_writer_mapping(auxiliary_problem.V)
            # ... auxiliary reduced dof map mappings
            key = (auxiliary_problem, component)
            if key not in self._auxiliary_reduced_dofs__dof_map_writer_mapping:
                self._auxiliary_reduced_dofs__dof_map_writer_mapping[key] = dict()
            if index not in self._auxiliary_reduced_dofs__dof_map_writer_mapping[key]:
                auxiliary_reduced_V = self._auxiliary_reduced_function_space[key][index]
                self._auxiliary_reduced_dofs__dof_map_writer_mapping[key][
                    index] = wrapping.build_dof_map_writer_mapping(auxiliary_reduced_V)

        def _save_auxiliary_basis_functions_matrix(self, directory, filename, auxiliary_problem, component, index):
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            full_directory.create()
            # Save auxiliary basis functions matrix
            logger.log(DEBUG, "Saving auxiliary reduced function space for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            key = (auxiliary_problem, component)
            auxiliary_basis_functions_matrix = self._auxiliary_basis_functions_matrix[key][index]
            full_directory_plus_key_and_index = Folders.Folder(
                os.path.join(str(full_directory), "auxiliary_basis_functions",
                             self._auxiliary_key_to_folder(key), str(index)))
            full_directory_plus_key_and_index.create()
            auxiliary_basis_functions_matrix.save(full_directory_plus_key_and_index, "auxiliary_basis")

        def _patch_auxiliary_basis_functions_matrix_save(self, directory, filename, auxiliary_problem, component):
            key = (auxiliary_problem, component)
            if not self._auxiliary_basis_functions_matrix_save_patched[key]:
                def update_and_save_auxiliary_basis_functions_matrix():
                    for index in range(len(self.reduced_mesh)):
                        self._update_auxiliary_basis_functions_matrix(auxiliary_problem, component, index)
                        self._save_auxiliary_basis_functions_matrix(
                            directory, filename, auxiliary_problem, component, index)

                auxiliary_reduction_method = get_reduction_method_from_problem(auxiliary_problem)
                is_pod_galerkin = hasattr(auxiliary_reduction_method, "compute_basis_functions")
                is_reduced_basis = hasattr(auxiliary_reduction_method, "update_basis_matrix")
                assert (is_pod_galerkin or is_reduced_basis) and not (is_pod_galerkin and is_reduced_basis)
                if is_pod_galerkin:
                    original_compute_basis_functions = auxiliary_reduction_method.compute_basis_functions

                    def patched_compute_basis_functions(self_):
                        original_compute_basis_functions()
                        update_and_save_auxiliary_basis_functions_matrix()

                    PatchInstanceMethod(auxiliary_reduction_method, "compute_basis_functions",
                                        patched_compute_basis_functions).patch()
                elif is_reduced_basis:
                    original_update_basis_matrix = auxiliary_reduction_method.update_basis_matrix

                    def patched_update_basis_matrix(self_, snapshot):
                        original_update_basis_matrix(snapshot)
                        update_and_save_auxiliary_basis_functions_matrix()

                    PatchInstanceMethod(auxiliary_reduction_method, "update_basis_matrix",
                                        patched_update_basis_matrix).patch()
                else:
                    raise TypeError("Unsupported reduction method")
                # Update bool value
                self._auxiliary_basis_functions_matrix_save_patched[key] = True

        def load(self, directory, filename):
            if len(self.reduced_mesh) > 0:  # avoid loading multiple times
                self._assert_dict_lengths()
                return False
            else:
                self._assert_dict_lengths()
                # Get full directory name
                full_directory = os.path.join(str(directory), filename)
                # Nmax
                Nmax = self._load_Nmax(directory, filename)
                # reduced_mesh
                for index in range(Nmax):
                    mesh_filename = os.path.join(str(directory), filename, "reduced_mesh_" + str(index))
                    with MeshFile(self.mesh.mpi_comm(), self.mesh.geometry().dim(), mesh_filename, "r") as input_file:
                        reduced_mesh = input_file.read()
                    self.reduced_mesh[index] = reduced_mesh
                    # Also initialize reduced function spaces
                    reduced_function_spaces = list()
                    for V_component in self.V:
                        reduced_function_space_component = wrapping.convert_functionspace_to_submesh(
                            V_component, reduced_mesh, self._get_reduced_function_space_type(V_component))
                        reduced_function_spaces.append(reduced_function_space_component)
                    self.reduced_function_spaces[index] = tuple(reduced_function_spaces)
                # reduced_subdomain_data
                for index in range(Nmax):
                    if self.subdomain_data is not None:
                        reduced_subdomain_data = dict()
                        for (subdomain_index, subdomain) in enumerate(self.subdomain_data):
                            subdomain_filename = os.path.join(
                                str(directory), filename, "reduced_mesh_" + str(index)
                                + "_subdomain_" + str(subdomain_index))
                            with MeshFunctionFile(self.mesh.mpi_comm(), self.mesh.geometry().dim(),
                                                  subdomain_filename, "r") as input_file:
                                reduced_subdomain = input_file.read(
                                    "size_t", self.reduced_mesh[index], subdomain.dim())
                            reduced_subdomain_data[subdomain] = reduced_subdomain
                        self.reduced_subdomain_data[index] = reduced_subdomain_data
                    else:
                        self.reduced_subdomain_data[index] = None
                # do not load reduced_mesh_markers, as they are not needed online
                # Init
                self._init_for_load_if_needed(Nmax)
                # reduced_mesh_dofs_list
                importable_reduced_mesh_dofs_list = ExportableList("pickle")
                importable_reduced_mesh_dofs_list.load(full_directory, "dofs")
                assert len(self.reduced_mesh_dofs_list) == 0
                importable_reduced_mesh_dofs_list__iterator = 0
                importable_reduced_mesh_dofs_list_tuple_length = len(self.V)
                while importable_reduced_mesh_dofs_list__iterator < len(importable_reduced_mesh_dofs_list):
                    reduced_mesh_dof = list()
                    for component in range(importable_reduced_mesh_dofs_list_tuple_length):
                        (global_cell_index, cell_dof) = (
                            importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator][0],
                            importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator][1])
                        reduced_mesh_dof.append(
                            self.reduced_mesh_dofs_list__dof_map_reader_mapping[component][global_cell_index][cell_dof])
                        importable_reduced_mesh_dofs_list__iterator += 1
                    self.reduced_mesh_dofs_list.append(tuple(reduced_mesh_dof))
                # reduced_mesh_reduced_dofs_list
                for index in range(Nmax):
                    importable_reduced_mesh_reduced_dofs_list = ExportableList("pickle")
                    importable_reduced_mesh_reduced_dofs_list.load(full_directory, "reduced_dofs_" + str(index))
                    assert len(self.reduced_mesh_reduced_dofs_list) == index
                    self.reduced_mesh_reduced_dofs_list[index] = list()
                    importable_reduced_mesh_reduced_dofs_list__iterator = 0
                    importable_reduced_mesh_reduced_dofs_list_tuple_length = len(self.V)
                    while importable_reduced_mesh_reduced_dofs_list__iterator < len(
                            importable_reduced_mesh_reduced_dofs_list):
                        reduced_mesh_dof = list()
                        for component in range(importable_reduced_mesh_reduced_dofs_list_tuple_length):
                            (global_cell_index, cell_dof) = (
                                importable_reduced_mesh_reduced_dofs_list[
                                    importable_reduced_mesh_reduced_dofs_list__iterator][0],
                                importable_reduced_mesh_reduced_dofs_list[
                                    importable_reduced_mesh_reduced_dofs_list__iterator][1])
                            reduced_mesh_dof.append(
                                self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping[
                                    index][component][global_cell_index][cell_dof])
                            importable_reduced_mesh_reduced_dofs_list__iterator += 1
                        self.reduced_mesh_reduced_dofs_list[index].append(tuple(reduced_mesh_dof))
                #
                self._assert_dict_lengths()

                # Auxiliary
                self._load_auxiliary(directory, filename)

                return True

        def _load_Nmax(self, directory, filename):
            def load_Nmax_task():
                with open(os.path.join(str(directory), filename, "reduced_mesh.length"), "r") as length:
                    return int(length.readline())
            return parallel_io(load_Nmax_task, self.mpi_comm)

        def _init_for_load_if_needed(self, Nmax):
            # Initialize dof map mappings for input
            if len(self.reduced_mesh_dofs_list__dof_map_reader_mapping) == 0:
                reduced_mesh_dofs_list__dof_map_reader_mapping = list()
                for V_component in self.V:
                    reduced_mesh_dofs_list__dof_map_reader_mapping.append(
                        wrapping.build_dof_map_reader_mapping(V_component))
                self.reduced_mesh_dofs_list__dof_map_reader_mapping = tuple(
                    reduced_mesh_dofs_list__dof_map_reader_mapping)

            # Initialize reduced dof map mappings for input
            for index in range(len(self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping), Nmax):
                reduced_mesh_reduced_dofs_list__dof_map_reader_mapping = list()
                for reduced_V__component in self.reduced_function_spaces[index]:
                    reduced_mesh_reduced_dofs_list__dof_map_reader_mapping.append(
                        wrapping.build_dof_map_reader_mapping(reduced_V__component))
                self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping[index] = tuple(
                    reduced_mesh_reduced_dofs_list__dof_map_reader_mapping)

        def _load_auxiliary(self, directory, filename):
            if self.auxiliary_problems_and_components is not None:
                for key in self.auxiliary_problems_and_components:
                    (auxiliary_problem, component) = key
                    for index in range(len(self.reduced_mesh)):
                        self._load_auxiliary_reduced_function_space(
                            directory, filename, auxiliary_problem, component, index)
                        if not isinstance(auxiliary_problem, wrapping.AuxiliaryProblemForNonParametrizedFunction):
                            if is_training_finished(auxiliary_problem):
                                self._load_auxiliary_basis_functions_matrix(
                                    directory, filename, auxiliary_problem, component, index)
                            else:
                                self._patch_auxiliary_basis_functions_matrix_load(
                                    directory, filename, auxiliary_problem, component)
                        # Re-create interpolator, as it was not saved to file
                        self._update_auxiliary_function_interpolator(auxiliary_problem, component, index)

        def _load_auxiliary_reduced_function_space(self, directory, filename, auxiliary_problem, component, index):
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            full_directory.create()
            # Create auxiliary reduced function space
            logger.log(DEBUG, "Loading auxiliary reduced function space for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            key = (auxiliary_problem, component)
            auxiliary_V = _sub_from_tuple(auxiliary_problem.V, component)
            auxiliary_reduced_V = wrapping.convert_functionspace_to_submesh(
                auxiliary_V, self.reduced_mesh[index], self._get_auxiliary_reduced_function_space_type(auxiliary_V))
            self._auxiliary_reduced_function_space[key][index] = auxiliary_reduced_V
            # Init
            self._init_for_auxiliary_load_if_needed(auxiliary_problem, component, index)
            # Load auxiliary dofs and reduced dofs
            importable_auxiliary_dofs = ExportableList("pickle")
            importable_auxiliary_reduced_dofs = ExportableList("pickle")
            full_directory_plus_key_and_index__dofs = Folders.Folder(
                os.path.join(str(full_directory), "auxiliary_dofs", self._auxiliary_key_to_folder(key), str(index)))
            full_directory_plus_key_and_index__reduced_dofs = Folders.Folder(
                os.path.join(str(full_directory), "auxiliary_reduced_dofs",
                             self._auxiliary_key_to_folder(key), str(index)))
            if (not full_directory_plus_key_and_index__dofs.create()
                    and not full_directory_plus_key_and_index__reduced_dofs.create()):
                importable_auxiliary_dofs.load(
                    full_directory_plus_key_and_index__dofs, "auxiliary_dofs")
                importable_auxiliary_reduced_dofs.load(
                    full_directory_plus_key_and_index__reduced_dofs, "auxiliary_reduced_dofs")
                auxiliary_dofs_to_reduced_dofs = dict()
                for (dof_input, reduced_dof_input) in zip(importable_auxiliary_dofs, importable_auxiliary_reduced_dofs):
                    dof = self._auxiliary_dofs__dof_map_reader_mapping[
                        auxiliary_problem][dof_input[0]][dof_input[1]]
                    reduced_dof = self._auxiliary_reduced_dofs__dof_map_reader_mapping[
                        key][index][reduced_dof_input[0]][reduced_dof_input[1]]
                    auxiliary_dofs_to_reduced_dofs[dof] = reduced_dof
                self._auxiliary_dofs_to_reduced_dofs[key][index] = auxiliary_dofs_to_reduced_dofs
            else:
                raise OSError

        def _init_for_auxiliary_load_if_needed(self, auxiliary_problem, component, index):
            # Initialize auxiliary dof map mappings and auxiliary reduced dof map mappings for input
            # ... auxiliary dof map mappings
            if auxiliary_problem not in self._auxiliary_dofs__dof_map_reader_mapping:
                self._auxiliary_dofs__dof_map_reader_mapping[
                    auxiliary_problem] = wrapping.build_dof_map_reader_mapping(auxiliary_problem.V)
            # ... auxiliary reduced dof map mappings
            key = (auxiliary_problem, component)
            if key not in self._auxiliary_reduced_dofs__dof_map_reader_mapping:
                self._auxiliary_reduced_dofs__dof_map_reader_mapping[key] = dict()
            if index not in self._auxiliary_reduced_dofs__dof_map_reader_mapping[key]:
                auxiliary_reduced_V = self._auxiliary_reduced_function_space[key][index]
                self._auxiliary_reduced_dofs__dof_map_reader_mapping[
                    key][index] = wrapping.build_dof_map_reader_mapping(auxiliary_reduced_V)

        def _load_auxiliary_basis_functions_matrix(self, directory, filename, auxiliary_problem, component, index):
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            full_directory.create()
            # Load auxiliary basis functions matrix
            logger.log(DEBUG, "Loading auxiliary basis functions matrix for " + auxiliary_problem.name()
                       + ", " + str(component) + ", " + str(index))
            key = (auxiliary_problem, component)
            full_directory_plus_key_and_index = Folders.Folder(
                os.path.join(str(full_directory), "auxiliary_basis_functions",
                             self._auxiliary_key_to_folder(key), str(index)))
            if not full_directory_plus_key_and_index.create():
                auxiliary_reduced_problem = get_reduced_problem_from_problem(auxiliary_problem)
                auxiliary_reduced_V = self.get_auxiliary_reduced_function_space(auxiliary_problem, component, index)
                auxiliary_basis_functions_matrix = self._init_auxiliary_basis_functions_matrix(
                    auxiliary_reduced_problem, component, auxiliary_reduced_V)
                auxiliary_basis_functions_matrix.load(full_directory_plus_key_and_index, "auxiliary_basis")
                self._auxiliary_basis_functions_matrix[key][index] = auxiliary_basis_functions_matrix
            else:
                raise OSError

        def _patch_auxiliary_basis_functions_matrix_load(self, directory, filename, auxiliary_problem, component):
            key = (auxiliary_problem, component)
            if not self._auxiliary_basis_functions_matrix_load_patched[key]:
                def load_auxiliary_basis_functions_matrix():
                    for index in range(len(self.reduced_mesh)):
                        if index not in self._auxiliary_basis_functions_matrix[key]:
                            self._load_auxiliary_basis_functions_matrix(
                                directory, filename, auxiliary_problem, component, index)

                auxiliary_reduction_method = get_reduction_method_from_problem(auxiliary_problem)
                original_finalize_offline = auxiliary_reduction_method._finalize_offline

                def patched_finalize_offline(self_):
                    original_finalize_offline()
                    load_auxiliary_basis_functions_matrix()

                PatchInstanceMethod(auxiliary_reduction_method, "_finalize_offline", patched_finalize_offline).patch()

                # Update bool value
                self._auxiliary_basis_functions_matrix_load_patched[key] = True

        @staticmethod
        @abstractmethod
        def _init_auxiliary_basis_functions_matrix(auxiliary_reduced_problem, components_tuple, auxiliary_reduced_V):
            pass

        def _auxiliary_key_to_folder(self, key):
            assert len(key) == 2
            (auxiliary_problem, component) = key
            folder_path = [auxiliary_problem.name()]
            assert isinstance(component, tuple)
            assert len(component) > 0
            if len(component) == 1:
                if component[0] is not None:
                    folder_path.append("component_" + str(component[0]))
            else:
                folder_path.append("component_" + "_".join([str(c) for c in component]))
            return os.path.join(*folder_path)

        def _assert_dict_lengths(self):
            assert len(self.reduced_mesh) == len(self.reduced_function_spaces)
            assert len(self.reduced_mesh) == len(self.reduced_subdomain_data)
            if len(self.reduced_mesh) == 0:
                assert len(self.reduced_mesh_dofs_list) == 0
            else:
                assert max(self.reduced_mesh.keys()) == len(self.reduced_mesh_dofs_list) - 1
            assert len(self.reduced_mesh) == len(self.reduced_mesh_reduced_dofs_list)

        def __getitem__(self, key):
            assert isinstance(key, slice)
            assert key.start is None
            assert key.step is None
            assert key.stop > 0
            output = _BasicReducedMesh.__new__(type(self), self.V, self.subdomain_data,
                                               self.auxiliary_problems_and_components,
                                               copy_from=self, key_as_slice=key, key_as_int=key.stop - 1)
            output.__init__(self.V, self.subdomain_data, self.auxiliary_problems_and_components,
                            copy_from=self, key_as_slice=key, key_as_int=key.stop - 1)
            return output

        def get_reduced_mesh(self, index=None):
            index = self._get_dict_index(index)
            return self.reduced_mesh[index]

        def get_reduced_function_spaces(self, index=None):
            index = self._get_dict_index(index)
            return self.reduced_function_spaces[index]

        def get_reduced_subdomain_data(self, index=None):
            index = self._get_dict_index(index)
            return self.reduced_subdomain_data[index]

        def get_dofs_list(self, index=None):
            index = self._get_dict_index(index)
            return self.reduced_mesh_dofs_list[:(index + 1)]  # increment so that slice will go up to index included

        def get_reduced_dofs_list(self, index=None):
            index = self._get_dict_index(index)
            return self.reduced_mesh_reduced_dofs_list[index]

        def get_auxiliary_reduced_function_space(self, auxiliary_problem, component, index=None):
            index = self._get_dict_index(index)
            return self._auxiliary_reduced_function_space[auxiliary_problem, component][index]

        def get_auxiliary_function_interpolator(self, auxiliary_problem, component, index=None):
            index = self._get_dict_index(index)
            return self._auxiliary_function_interpolator[auxiliary_problem, component][index]

        def get_auxiliary_basis_functions_matrix(self, auxiliary_problem, component, index=None):
            index = self._get_dict_index(index)
            return self._auxiliary_basis_functions_matrix[auxiliary_problem, component][index]

        def _get_dict_index(self, index):
            self._assert_dict_lengths()
            if index is None:
                return max(self.reduced_mesh.keys())
            else:
                return index

        def _get_next_index(self):
            N = len(self.reduced_mesh)
            if N > 0:
                assert min(self.reduced_mesh.keys()) == 0
                assert max(self.reduced_mesh.keys()) == N - 1
            return N

    return _BasicReducedMesh


backend = ModuleWrapper(BasisFunctionsMatrix)
wrapping = ModuleWrapper(AuxiliaryProblemForNonParametrizedFunction, build_dof_map_reader_mapping,
                         build_dof_map_writer_mapping, create_submesh, convert_meshfunctions_to_submesh,
                         convert_functionspace_to_submesh, evaluate_sparse_function_at_dofs,
                         map_functionspaces_between_mesh_and_submesh,
                         evaluate_basis_functions_matrix_at_dofs=evaluate_basis_functions_matrix_at_dofs)
ReducedMesh_Base = BasicReducedMesh(backend, wrapping)


@BackendFor("dolfin", inputs=(FunctionSpace, ))
class ReducedMesh(ReducedMesh_Base):
    def _compute_dof_to_cells(self, V_component):
        assert isinstance(V_component, FunctionSpace)
        dof_to_cells = dict()  # from global dof to cell
        for cell in cells(self.mesh):
            local_dofs = V_component.dofmap().cell_dofs(cell.index())
            for local_dof in local_dofs:
                global_dof = V_component.dofmap().local_to_global_index(local_dof)
                if global_dof not in dof_to_cells:
                    dof_to_cells[global_dof] = list()
                if cell not in dof_to_cells[global_dof]:
                    dof_to_cells[global_dof].append(cell)
        return dof_to_cells

    @staticmethod
    def _get_reduced_function_space_type(V_component):
        assert isinstance(V_component, FunctionSpace)
        if hasattr(V_component, "_component_to_index"):
            def CustomFunctionSpace(mesh, element):
                return FunctionSpace(mesh, element, components=V_component._component_to_index)
            return CustomFunctionSpace
        else:
            return FunctionSpace

    @staticmethod
    def _init_auxiliary_basis_functions_matrix(auxiliary_reduced_problem, components_tuple, auxiliary_reduced_V):
        auxiliary_basis_functions_matrix = backend.BasisFunctionsMatrix(auxiliary_reduced_V)
        assert isinstance(components_tuple, tuple)
        assert len(components_tuple) > 0
        if len(components_tuple) > 1:
            # This handles the case where a subcomponent of a component is required
            # (e.g., x subcomponent of the velocity field component of a (velocity, pressure) solution)
            # Since basis are constructed with respect to components (rather than subcomponents) we
            # use only the first entry in the tuple to detect the corresponding component name
            assert all([isinstance(c, int) for c in components_tuple])  # there is no None and all entries are integer
        component_as_int = components_tuple[0]
        if component_as_int is None:  # all components
            # Initialize a basis function matrix for all components
            components_name = auxiliary_reduced_problem.basis_functions._components_name
        elif len(auxiliary_reduced_problem.basis_functions._components_name
                 ) == 1:  # subcomponent of a problem with only one component
            # Initialize a basis function matrix for all components
            components_name = auxiliary_reduced_problem.basis_functions._components_name
        else:
            # Initialize a basis function matrix only for the required integer component
            if len(auxiliary_reduced_V._index_to_components) == 1:
                components_name = auxiliary_reduced_V.index_to_components(None)
            else:
                assert isinstance(component_as_int, int)
                components_name = auxiliary_reduced_V.index_to_components(component_as_int)
        auxiliary_basis_functions_matrix.init(components_name)
        return auxiliary_basis_functions_matrix

    @staticmethod
    def _get_auxiliary_reduced_function_space_type(auxiliary_V):
        assert isinstance(auxiliary_V, FunctionSpace)
        if hasattr(auxiliary_V, "_component_to_index"):
            def CustomFunctionSpace(mesh, element):
                return FunctionSpace(mesh, element, components=auxiliary_V._component_to_index)
            return CustomFunctionSpace
        else:
            return FunctionSpace
