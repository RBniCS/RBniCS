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

from dolfin import CellFunction, cells, DEBUG, entities, File, has_hdf5, has_hdf5_parallel, log, Mesh, MeshFunction, XDMFFile
if has_hdf5() and has_hdf5_parallel():
    from dolfin import HDF5File
    hdf5_file_type = "h5" # Will be switched to "xdmf" in future, because there is currently a bug in reading back in 1D meshes from XDMF
import rbnics.backends.dolfin
from rbnics.backends.abstract import ReducedMesh as AbstractReducedMesh
from rbnics.backends.dolfin.wrapping import FunctionSpace
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import _sub_from_tuple
from rbnics.utils.decorators import BackendFor, get_problem_from_problem_name
from rbnics.utils.io import ExportableList, Folders
from rbnics.utils.mpi import is_io_process
from mpi4py.MPI import MAX, SUM

@BackendFor("dolfin", inputs=(FunctionSpace, ))
class ReducedMesh(AbstractReducedMesh):
    def __init__(self, V, subdomain_data=None, **kwargs):
        AbstractReducedMesh.__init__(self, V)
        #
        assert isinstance(V, tuple)
        assert len(V) in (1, 2)
        if len(V) == 2:
            assert V[0].mesh().ufl_domain() == V[1].mesh().ufl_domain()
        self.mesh = V[0].mesh()
        self.mpi_comm = self.mesh.mpi_comm().tompi4py()
        self.V = V
        self.subdomain_data = subdomain_data
        
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
            
        # Detect if a custom backend has been provided in kwargs
        if "backend" in kwargs:
            self.backend = kwargs["backend"]
        else:
            self.backend = rbnics.backends.dolfin
            
        # Prepare storage for an helper dof to cell dict
        self.dof_to_cells = tuple() # of size len(V)
        # ... which is not initialized in the constructor to avoid wasting time online
        # ... since it is only needed offline in the append() method
        
        # Cell functions to mark cells (on the full mesh)
        self.reduced_mesh_markers = dict() # from N to CellFunction
        # ... which again is not initialized here for performance reasons
        
        # DOFs list (of the full mesh) that need to be added at each N
        self.reduced_mesh_dofs_list = list() # list (of size N) of tuple (of size len(V)) of dofs
        if copy_from is not None:
            self.reduced_mesh_dofs_list.extend(copy_from.reduced_mesh_dofs_list[key_as_slice])
        # Prepare storage for helper mapping needed for I/O
        self.reduced_mesh_dofs_list__dof_map_writer_mapping = tuple() # of size len(V)
        self.reduced_mesh_dofs_list__dof_map_reader_mapping = tuple() # of size len(V)
        # ... which will be initialized as needed in the save and load methods
                
        # Reduced meshes, for all N
        self.reduced_mesh = dict() # from N to Mesh
        if copy_from is not None:
            self.reduced_mesh[key_as_int] = copy_from.reduced_mesh[key_as_int]
            
        # Reduced subdomain data, for all N
        self.reduced_subdomain_data = dict() # from N to dict from mesh MeshFunction to reduced_mesh MeshFunction
        if copy_from is not None:
            self.reduced_subdomain_data[key_as_int] = copy_from.reduced_subdomain_data[key_as_int]
            
        # Reduced function spaces, for all N
        self.reduced_function_spaces = dict() # from N to tuple (of size len(V)) of FunctionSpace
        if copy_from is not None:
            self.reduced_function_spaces[key_as_int] = copy_from.reduced_function_spaces[key_as_int]
            
        # DOFs list (of the reduced mesh) that need to be added at each N
        self.reduced_mesh_reduced_dofs_list = dict() # from N to list of tuple (of size len(V)) of dofs
        if copy_from is not None:
            self.reduced_mesh_reduced_dofs_list[key_as_int] = copy_from.reduced_mesh_reduced_dofs_list[key_as_int]
        # Prepare storage for helper mapping needed for I/O
        self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping = dict() # from N to tuple (of size len(V))
        self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping = dict() # from N to tuple (of size len(V))
        # ... which will be initialized as needed in the save and load methods
        
        ## The following members are related to auxiliary basis functions for nonlinear terms.
        # Spaces for auxiliary basis functions
        self._auxiliary_reduced_function_space = dict() # from (problem, N) to FunctionSpace
        if copy_from is not None:
            self._auxiliary_reduced_function_space = copy_from._auxiliary_reduced_function_space
        # Mapping between DOFs on the reduced mesh and DOFs on the full mesh for auxiliary basis functions
        self._auxiliary_dofs_to_reduced_dofs = dict() # from (problem, N) to dict from int to int
        if copy_from is not None:
            self._auxiliary_dofs_to_reduced_dofs = copy_from._auxiliary_dofs_to_reduced_dofs
        # Auxiliary basis functions
        self._auxiliary_basis_functions_matrix = dict() # from (problem, N) to BasisFunctionsMatrix
        if copy_from is not None:
            self._auxiliary_basis_functions_matrix = copy_from._auxiliary_basis_functions_matrix
        # Auxiliary function interpolator
        self._auxiliary_function_interpolator = dict() # from (problem, N) to function
        if copy_from is not None:
            self._auxiliary_function_interpolator = copy_from._auxiliary_function_interpolator
        # Prepare storage for helper mapping needed for I/O
        self._auxiliary_dofs__dof_map_writer_mapping = dict() # from problem
        self._auxiliary_dofs__dof_map_reader_mapping = dict() # from problem
        self._auxiliary_reduced_dofs__dof_map_writer_mapping = dict() # from (problem, N)
        self._auxiliary_reduced_dofs__dof_map_reader_mapping = dict() # from (problem, N)
        # ... which will be initialized as needed in the save and load methods
        # Store directory and filename passed to save()
        self._auxiliary_io_directory = None
        self._auxiliary_io_filename = None
        if copy_from is not None:
            self._auxiliary_io_directory = copy_from._auxiliary_io_directory
            self._auxiliary_io_filename = copy_from._auxiliary_io_filename
        
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
        reduced_mesh = self.backend.wrapping.create_submesh(self.mesh, self.reduced_mesh_markers[N])
        self.reduced_mesh[N] = reduced_mesh
        # Create subdomain data on submesh
        if self.subdomain_data is not None:
            reduced_subdomain_data_list = self.backend.wrapping.convert_meshfunctions_to_submesh(self.mesh, reduced_mesh, self.subdomain_data)
            reduced_subdomain_data = dict()
            assert len(self.subdomain_data) == len(reduced_subdomain_data_list)
            for (subdomain, reduced_subdomain) in zip(self.subdomain_data, reduced_subdomain_data_list):
                reduced_subdomain_data[subdomain] = reduced_subdomain
            self.reduced_subdomain_data[N] = reduced_subdomain_data
        else:
            self.reduced_subdomain_data[N] = None
        # Store the FunctionSpace V on the reduced mesh, as well as the map between DOFs on V and reduced_V
        reduced_function_spaces = list()
        dofs__to__reduced_dofs = list() # of size len(V)
        for (component, V_component) in enumerate(self.V):
            reduced_function_space_component = self.backend.wrapping.convert_functionspace_to_submesh(V_component, reduced_mesh, self._get_reduced_function_space_type(V_component))
            reduced_function_spaces.append(reduced_function_space_component)
            (dofs__to__reduced_dofs_component, _) = self.backend.wrapping.map_functionspaces_between_mesh_and_submesh(V_component, self.mesh, reduced_function_space_component, reduced_mesh)
            dofs__to__reduced_dofs.append(dofs__to__reduced_dofs_component)
            log(DEBUG, "DOFs to reduced DOFs (component " + str(component) +") is " + str(dofs__to__reduced_dofs[component]))
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
        log(DEBUG, "Reduced DOFs list " + str(reduced_mesh_reduced_dofs_list))
        log(DEBUG, "corresponding to DOFs list " + str(self.reduced_mesh_dofs_list))
        self.reduced_mesh_reduced_dofs_list[N] = reduced_mesh_reduced_dofs_list
        
    def _init_for_append_if_needed(self):
        # Initialize dof to cells map only the first time
        if len(self.dof_to_cells) == 0:
            self.dof_to_cells = list() # of size len(V)
            for (component, V_component) in enumerate(self.V):
                dof_to_cells = self._compute_dof_to_cells(V_component)
                # Debugging
                log(DEBUG, "DOFs to cells map (component " + str(component) +") on processor " + str(self.mpi_comm.rank) + ":")
                for (global_dof, cells_) in dof_to_cells.items():
                    log(DEBUG, "\t" + str(global_dof) + ": " + str([cell.global_index() for cell in cells_]))
                # Add to storage
                self.dof_to_cells.append(dof_to_cells)
            self.dof_to_cells = tuple(self.dof_to_cells)
        # Initialize cells marker
        N = self._get_next_index()
        reduced_mesh_markers = CellFunction("bool", self.mesh)
        reduced_mesh_markers.set_all(False)
        if N > 0:
            reduced_mesh_markers.array()[:] = self.reduced_mesh_markers[N - 1].array()
        assert N not in self.reduced_mesh_markers
        self.reduced_mesh_markers[N] = reduced_mesh_markers
        
    def _compute_dof_to_cells(self, V_component):
        dof_to_cells = dict() # from global dof to cell
        for cell in cells(self.mesh):
            local_dofs = V_component.dofmap().cell_dofs(cell.index())
            for local_dof in local_dofs:
                global_dof = V_component.dofmap().local_to_global_index(local_dof)
                if not global_dof in dof_to_cells:
                    dof_to_cells[global_dof] = list()
                if not cell in dof_to_cells[global_dof]:
                    dof_to_cells[global_dof].append(cell)
        return dof_to_cells
        
    def _get_reduced_function_space_type(self, V_component):
        if hasattr(V_component, "_component_to_index"):
            def CustomFunctionSpace(mesh, element):
                return FunctionSpace(mesh, element, components=V_component._component_to_index)
            return CustomFunctionSpace
        else:
            return FunctionSpace
    
    def save(self, directory, filename):
        self._assert_dict_lengths()
        # Get full directory name
        full_directory = Folders.Folder(directory + "/" + filename)
        full_directory.create()
        # Nmax
        self._save_Nmax(directory, filename)
        # reduced_mesh
        for (index, reduced_mesh) in self.reduced_mesh.items():
            mesh_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index)
            if not has_hdf5() or not has_hdf5_parallel():
                assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
                mesh_filename = mesh_filename + ".xml"
                File(mesh_filename) << reduced_mesh
            else:
                assert hdf5_file_type in ("h5", "xdmf")
                if hdf5_file_type == "h5":
                    mesh_filename = mesh_filename + ".h5"
                    output_file = HDF5File(self.mesh.mpi_comm(), mesh_filename, "w")
                    output_file.write(reduced_mesh, "/mesh")
                    output_file.close()
                else:
                    mesh_filename = mesh_filename + ".xdmf"
                    output_file = XDMFFile(self.mesh.mpi_comm(), mesh_filename)
                    output_file.write(reduced_mesh)
                    output_file.close()
        # cannot save reduced_function_spaces to file
        # reduced_subdomain_data
        if self.subdomain_data is not None:
            for (index, reduced_subdomain_data) in self.reduced_subdomain_data.items():
                subdomain_index = 0
                for (subdomain, reduced_subdomain) in reduced_subdomain_data.items():
                    subdomain_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index) + "_subdomain_" + str(subdomain_index)
                    if not has_hdf5() or not has_hdf5_parallel():
                        assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh function in parallel"
                        subdomain_filename = subdomain_filename + ".xml"
                        File(subdomain_filename) << reduced_subdomain
                    else:
                        assert hdf5_file_type in ("h5", "xdmf")
                        if hdf5_file_type == "h5":
                            subdomain_filename = subdomain_filename + ".h5"
                            output_file = HDF5File(self.mesh.mpi_comm(), subdomain_filename, "w")
                            output_file.write(reduced_subdomain, "/subdomain")
                            output_file.close()
                        else:
                            subdomain_filename = subdomain_filename + ".xdmf"
                            output_file = XDMFFile(self.mesh.mpi_comm(), subdomain_filename)
                            output_file.write(reduced_subdomain)
                            output_file.close()
                    subdomain_index += 1
        # reduced_mesh_markers
        for (index, reduced_mesh_markers) in self.reduced_mesh_markers.items():
            marker_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index) + "_markers"
            if not has_hdf5() or not has_hdf5_parallel():
                assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh function in parallel"
                marker_filename = marker_filename + ".xml"
                File(marker_filename) << reduced_mesh_markers
            else:
                assert hdf5_file_type in ("h5", "xdmf")
                if hdf5_file_type == "h5":
                    marker_filename = marker_filename + ".h5"
                    output_file = HDF5File(self.mesh.mpi_comm(), marker_filename, "w")
                    output_file.write(reduced_mesh_markers, "/markers")
                    output_file.close()
                else:
                    marker_filename = marker_filename + ".xdmf"
                    output_file = XDMFFile(self.mesh.mpi_comm(), marker_filename)
                    output_file.write(reduced_mesh_markers)
                    output_file.close()
        # Init
        self._init_for_save_if_needed()
        # reduced_mesh_dofs_list
        exportable_reduced_mesh_dofs_list = ExportableList("pickle")
        for reduced_mesh_dof in self.reduced_mesh_dofs_list:
            for (component, reduced_mesh_dof__component) in enumerate(reduced_mesh_dof):
                exportable_reduced_mesh_dofs_list.append(self.reduced_mesh_dofs_list__dof_map_writer_mapping[component][reduced_mesh_dof__component])
        exportable_reduced_mesh_dofs_list.save(full_directory, "dofs")
        # reduced_mesh_reduced_dofs_list
        for (index, reduced_mesh_reduced_dofs_list) in self.reduced_mesh_reduced_dofs_list.items():
            exportable_reduced_mesh_reduced_dofs_list = ExportableList("pickle")
            for reduced_mesh_reduced_dof in reduced_mesh_reduced_dofs_list:
                for (component, reduced_mesh_reduced_dof__component) in enumerate(reduced_mesh_reduced_dof):
                    exportable_reduced_mesh_reduced_dofs_list.append(self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping[index][component][reduced_mesh_reduced_dof__component])
            exportable_reduced_mesh_reduced_dofs_list.save(full_directory, "reduced_dofs_" + str(index))
            
        ## Auxiliary basis functions
        # We will not save anything, because saving to file is handled by get_auxiliary_* methods.
        # We need howewer to store the directory and filename where to save
        if self._auxiliary_io_directory is None:
            self._auxiliary_io_directory = directory
        else:
            assert self._auxiliary_io_directory == directory
        if self._auxiliary_io_filename is None:
            self._auxiliary_io_filename = filename
        else:
            assert self._auxiliary_io_filename == filename
            
    def _save_Nmax(self, directory, filename):
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + "/" + "reduced_mesh.length", "w") as length:
                length.write(str(len(self.reduced_mesh)))
        self.mpi_comm.barrier()
        
    def _save_auxiliary_reduced_function_space(self, key):
        # Get full directory name
        full_directory = Folders.Folder(self._auxiliary_io_directory + "/" + self._auxiliary_io_filename)
        full_directory.create()
        # Init
        self._init_for_auxiliary_save_if_needed()
        # Save auxiliary dofs and reduced dofs
        auxiliary_dofs_to_reduced_dofs = self._auxiliary_dofs_to_reduced_dofs[key]
        # ... auxiliary dofs
        exportable_auxiliary_dofs = ExportableList("pickle")
        for auxiliary_dof in auxiliary_dofs_to_reduced_dofs.keys():
            exportable_auxiliary_dofs.append(self._auxiliary_dofs__dof_map_writer_mapping[key[0]][auxiliary_dof])
        full_directory_plus_key__dofs = Folders.Folder(full_directory + "/auxiliary_dofs/" + self._auxiliary_key_to_folder(key))
        full_directory_plus_key__dofs.create()
        exportable_auxiliary_dofs.save(full_directory_plus_key__dofs, "auxiliary_dofs")
        # ... auxiliary reduced dofs
        exportable_auxiliary_reduced_dofs = ExportableList("pickle")
        for auxiliary_reduced_dof in auxiliary_dofs_to_reduced_dofs.values():
            exportable_auxiliary_reduced_dofs.append(self._auxiliary_reduced_dofs__dof_map_writer_mapping[key][auxiliary_reduced_dof])
        full_directory_plus_key__reduced_dofs = Folders.Folder(full_directory + "/auxiliary_reduced_dofs/" + self._auxiliary_key_to_folder(key))
        full_directory_plus_key__reduced_dofs.create()
        exportable_auxiliary_reduced_dofs.save(full_directory_plus_key__reduced_dofs, "auxiliary_reduced_dofs")
            
    def _save_auxiliary_basis_functions_matrix(self, key):
        # Get full directory name
        full_directory = Folders.Folder(self._auxiliary_io_directory + "/" + self._auxiliary_io_filename)
        full_directory.create()
        # Save auxiliary basis functions matrix
        auxiliary_basis_functions_matrix = self._auxiliary_basis_functions_matrix[key]
        full_directory_plus_key = Folders.Folder(full_directory + "/auxiliary_basis_functions/" + self._auxiliary_key_to_folder(key))
        full_directory_plus_key.create()
        auxiliary_basis_functions_matrix.save(full_directory_plus_key, "auxiliary_basis")
            
    def _init_for_save_if_needed(self):
        # Initialize dof map mappings for output
        if len(self.reduced_mesh_dofs_list__dof_map_writer_mapping) == 0:
            reduced_mesh_dofs_list__dof_map_writer_mapping = list()
            for V_component in self.V:
                reduced_mesh_dofs_list__dof_map_writer_mapping.append( self.backend.wrapping.build_dof_map_writer_mapping(V_component) )
            self.reduced_mesh_dofs_list__dof_map_writer_mapping = tuple(reduced_mesh_dofs_list__dof_map_writer_mapping)
            
        # Initialize reduced dof mapping for output
        assert len(self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping) == len(self.reduced_mesh) - 1
        reduced_mesh_reduced_dofs_list__dof_map_writer_mapping = list()
        for reduced_V__component in self.reduced_function_spaces[len(self.reduced_mesh) - 1]:
            reduced_mesh_reduced_dofs_list__dof_map_writer_mapping.append( self.backend.wrapping.build_dof_map_writer_mapping(reduced_V__component) )
        self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping[len(self.reduced_mesh) - 1] = tuple(reduced_mesh_reduced_dofs_list__dof_map_writer_mapping)
            
    def _init_for_auxiliary_save_if_needed(self):
        # Initialize auxiliary dof map mappings and auxiliary reduced dof map mappings for output
        for (key, auxiliary_reduced_V) in self._auxiliary_reduced_function_space.items():
            # auxiliary dof map mappings
            auxiliary_problem = key[0]
            if auxiliary_problem not in self._auxiliary_dofs__dof_map_writer_mapping:
                self._auxiliary_dofs__dof_map_writer_mapping[auxiliary_problem] = self.backend.wrapping.build_dof_map_writer_mapping(auxiliary_problem.V)
            # auxiliary reduced dof map mappings
            if key not in self._auxiliary_reduced_dofs__dof_map_writer_mapping:
                self._auxiliary_reduced_dofs__dof_map_writer_mapping[key] = self.backend.wrapping.build_dof_map_writer_mapping(auxiliary_reduced_V)
    
    def load(self, directory, filename):
        if len(self.reduced_mesh) > 0: # avoid loading multiple times
            self._assert_dict_lengths()
            return False
        else:
            self._assert_dict_lengths()
            # Get full directory name
            full_directory = directory + "/" + filename
            # Nmax
            Nmax = self._load_Nmax(directory, filename)
            # reduced_mesh
            for index in range(Nmax):
                mesh_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index)
                if not has_hdf5() or not has_hdf5_parallel():
                    assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to load a mesh in parallel"
                    mesh_filename = mesh_filename + ".xml"
                    reduced_mesh = Mesh(mesh_filename)
                else:
                    reduced_mesh = Mesh()
                    assert hdf5_file_type in ("h5", "xdmf")
                    if hdf5_file_type == "h5":
                        mesh_filename = mesh_filename + ".h5"
                        input_file = HDF5File(self.mesh.mpi_comm(), mesh_filename, "r")
                        input_file.read(reduced_mesh, "/mesh", False)
                        input_file.close()
                    else:
                        mesh_filename = mesh_filename + ".xdmf"
                        input_file = XDMFFile(self.mesh.mpi_comm(), mesh_filename)
                        input_file.read(reduced_mesh)
                        input_file.close()
                self.reduced_mesh[index] = reduced_mesh
                # Also initialize reduced function spaces
                reduced_function_spaces = list()
                for V_component in self.V:
                    reduced_function_space_component = self.backend.wrapping.convert_functionspace_to_submesh(V_component, reduced_mesh, self._get_reduced_function_space_type(V_component))
                    reduced_function_spaces.append(reduced_function_space_component)
                self.reduced_function_spaces[index] = tuple(reduced_function_spaces)
            # reduced_subdomain_data
            for index in range(Nmax):
                if self.subdomain_data is not None:
                    reduced_subdomain_data = dict()
                    for (subdomain_index, subdomain) in enumerate(self.subdomain_data):
                        subdomain_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index) + "_subdomain_" + str(subdomain_index)
                        if not has_hdf5() or not has_hdf5_parallel():
                            assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to load a mesh in parallel"
                            subdomain_filename = subdomain_filename + ".xml"
                            reduced_subdomain = MeshFunction("size_t", self.reduced_mesh[index], subdomain_filename)
                        else:
                            reduced_subdomain = MeshFunction("size_t", self.reduced_mesh[index], subdomain.dim())
                            assert hdf5_file_type in ("h5", "xdmf")
                            if hdf5_file_type == "h5":
                                subdomain_filename = subdomain_filename + ".xdmf"
                                input_file = XDMFFile(self.mesh.mpi_comm(), subdomain_filename)
                                input_file.read(reduced_subdomain)
                                input_file.close()
                            else:
                                subdomain_filename = subdomain_filename + ".h5"
                                input_file = HDF5File(self.mesh.mpi_comm(), subdomain_filename, "r")
                                input_file.read(reduced_subdomain)
                                input_file.close()
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
                    (global_cell_index, cell_dof) = (importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator][0], importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator][1])
                    reduced_mesh_dof.append( self.reduced_mesh_dofs_list__dof_map_reader_mapping[component][global_cell_index][cell_dof] )
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
                while importable_reduced_mesh_reduced_dofs_list__iterator < len(importable_reduced_mesh_reduced_dofs_list):
                    reduced_mesh_dof = list()
                    for component in range(importable_reduced_mesh_reduced_dofs_list_tuple_length):
                        (global_cell_index, cell_dof) = (importable_reduced_mesh_reduced_dofs_list[importable_reduced_mesh_reduced_dofs_list__iterator][0], importable_reduced_mesh_reduced_dofs_list[importable_reduced_mesh_reduced_dofs_list__iterator][1])
                        reduced_mesh_dof.append( self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping[index][component][global_cell_index][cell_dof] )
                        importable_reduced_mesh_reduced_dofs_list__iterator += 1
                    self.reduced_mesh_reduced_dofs_list[index].append(tuple(reduced_mesh_dof))
            #
            self._assert_dict_lengths()
            
            ## Auxiliary basis functions
            # Store the directory and filename where to load (and possibily save _get_auxiliary_*
            # quantities which were not computed already)
            if self._auxiliary_io_directory is None:
                self._auxiliary_io_directory = directory
            else:
                assert self._auxiliary_io_directory == directory
            if self._auxiliary_io_filename is None:
                self._auxiliary_io_filename = filename
            else:
                assert self._auxiliary_io_filename == filename
            
            return True
            
    def _init_for_load_if_needed(self, Nmax):
        # Initialize dof map mappings for input
        if len(self.reduced_mesh_dofs_list__dof_map_reader_mapping) == 0:
            reduced_mesh_dofs_list__dof_map_reader_mapping = list()
            for V_component in self.V:
                reduced_mesh_dofs_list__dof_map_reader_mapping.append( self.backend.wrapping.build_dof_map_reader_mapping(V_component) )
            self.reduced_mesh_dofs_list__dof_map_reader_mapping = tuple(reduced_mesh_dofs_list__dof_map_reader_mapping)
            
        # Initialize reduced dof map mappings for input
        for index in range(len(self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping), Nmax):
            reduced_mesh_reduced_dofs_list__dof_map_reader_mapping = list()
            for reduced_V__component in self.reduced_function_spaces[index]:
                reduced_mesh_reduced_dofs_list__dof_map_reader_mapping.append( self.backend.wrapping.build_dof_map_reader_mapping(reduced_V__component) )
            self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping[index] = tuple(reduced_mesh_reduced_dofs_list__dof_map_reader_mapping)
            
    def _init_for_auxiliary_load_if_needed(self):
        # Initialize auxiliary dof map mappings and auxiliary reduced dof map mappings for input
        for (key, auxiliary_reduced_V) in self._auxiliary_reduced_function_space.items():
            # auxiliary dof map mappings
            auxiliary_problem = key[0]
            if auxiliary_problem not in self._auxiliary_dofs__dof_map_reader_mapping:
                self._auxiliary_dofs__dof_map_reader_mapping[auxiliary_problem] = self.backend.wrapping.build_dof_map_reader_mapping(auxiliary_problem.V)
            # auxiliary reduced dof map mappings
            if key not in self._auxiliary_reduced_dofs__dof_map_reader_mapping:
                self._auxiliary_reduced_dofs__dof_map_reader_mapping[key] = self.backend.wrapping.build_dof_map_reader_mapping(auxiliary_reduced_V)
    
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + "/" + "reduced_mesh.length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self.mpi_comm.bcast(Nmax, root=is_io_process.root)
        return Nmax
        
    def _load_auxiliary_reduced_function_space(self, key):
        # Get full directory name
        full_directory = Folders.Folder(self._auxiliary_io_directory + "/" + self._auxiliary_io_filename)
        directory_exists = full_directory.create()
        # Init
        self._init_for_auxiliary_load_if_needed()
        # Load auxiliary dofs and reduced dofs
        importable_auxiliary_dofs = ExportableList("pickle")
        importable_auxiliary_reduced_dofs = ExportableList("pickle")
        full_directory_plus_key__dofs = Folders.Folder(full_directory + "/auxiliary_dofs/" + self._auxiliary_key_to_folder(key))
        full_directory_plus_key__reduced_dofs = Folders.Folder(full_directory + "/auxiliary_reduced_dofs/" + self._auxiliary_key_to_folder(key))
        if not full_directory_plus_key__dofs.create() and not full_directory_plus_key__reduced_dofs.create():
            importable_auxiliary_dofs.load(full_directory_plus_key__dofs, "auxiliary_dofs")
            importable_auxiliary_reduced_dofs.load(full_directory_plus_key__reduced_dofs, "auxiliary_reduced_dofs")
            auxiliary_dofs_to_reduced_dofs = dict()
            for (dof_input, reduced_dof_input) in zip(importable_auxiliary_dofs, importable_auxiliary_reduced_dofs):
                dof = self._auxiliary_dofs__dof_map_reader_mapping[key[0]][dof_input[0]][dof_input[1]]
                reduced_dof = self._auxiliary_reduced_dofs__dof_map_reader_mapping[key][reduced_dof_input[0]][reduced_dof_input[1]]
                auxiliary_dofs_to_reduced_dofs[dof] = reduced_dof
            self._auxiliary_dofs_to_reduced_dofs[key] = auxiliary_dofs_to_reduced_dofs
            return True
        else:
            return False
                
    def _load_auxiliary_basis_functions_matrix(self, key, auxiliary_reduced_problem, auxiliary_reduced_V):
        # Get full directory name
        full_directory = Folders.Folder(self._auxiliary_io_directory + "/" + self._auxiliary_io_filename)
        full_directory.create()
        # Load auxiliary basis functions matrix
        full_directory_plus_key = Folders.Folder(full_directory + "/auxiliary_basis_functions/" + self._auxiliary_key_to_folder(key))
        if not full_directory_plus_key.create():
            auxiliary_basis_functions_matrix = self._init_auxiliary_basis_functions_matrix(key, auxiliary_reduced_problem, auxiliary_reduced_V)
            auxiliary_basis_functions_matrix.load(full_directory_plus_key, "auxiliary_basis")
            self._auxiliary_basis_functions_matrix[key] = auxiliary_basis_functions_matrix
            return True
        else:
            return False
            
    def _init_auxiliary_basis_functions_matrix(self, key, auxiliary_reduced_problem, auxiliary_reduced_V):
        auxiliary_basis_functions_matrix = self.backend.BasisFunctionsMatrix(auxiliary_reduced_V)
        components_tuple = key[1]
        assert isinstance(components_tuple, tuple)
        assert len(components_tuple) > 0
        if len(components_tuple) == 1:
            component_as_int = components_tuple[0]
            if (
                component_as_int is None # all components
                    or 
                len(auxiliary_reduced_problem.Z._components_name) is 1 # subcomponent of a problem with only one component
            ):
                # Initialize a basis function matrix for all components
                components_name = auxiliary_reduced_problem.Z._components_name
            else:
                # Initialize a basis function matrix only for the required integer component
                assert isinstance(component_as_int, int)
                components_name = [auxiliary_reduced_problem.Z._components_name[component_as_int]]
        else:
            # This handles the case where a subcomponent of a component is required
            # (e.g., x subcomponent of the velocity field component of a (velocity, pressure) solution)
            # Since basis are constructed with respect to components (rather than subcomponents) we
            # use only the first entry in the tuple to detect the corresponding component name
            assert all([isinstance(c, int) for c in components_tuple]) # there is no None and all entries are integer
            if len(auxiliary_reduced_problem.Z._components_name) is 1: # subcomponent of a problem with only one component
                components_name = [auxiliary_reduced_problem.Z._components_name[0]]
            else: # subcomponent of a problem with more than one component
                component_as_int = components_tuple[0]
                components_name = [auxiliary_reduced_problem.Z._components_name[component_as_int]]
            # Note that the discard of all other subcomponents is automatically handled by
            # evaluate_basis_functions_matrix_at_dofs, which will be provided reduced dofs
            # acting only on the active subcomponent.
        auxiliary_basis_functions_matrix.init(components_name)
        return auxiliary_basis_functions_matrix
        
    def _auxiliary_key_to_folder(self, key):
        folder = key[0].name() + "/"
        assert isinstance(key[1], tuple)
        assert len(key[1]) > 0
        if len(key[1]) is 1:
            if key[1][0] is not None:
                folder += "component_" + str(key[1][0]) + "/"
        else:
            folder += "component_" + "_".join([str(c) for c in key[1]]) + "/"
        folder += str(key[2])
        return folder
            
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
        return self.backend.ReducedMesh(self.V, self.subdomain_data, copy_from=self, key_as_slice=key, key_as_int=key.stop - 1, backend=self.backend)
                
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
        return self.reduced_mesh_dofs_list[:(index + 1)] # increment so that slice will go up to index included
        
    def get_reduced_dofs_list(self, index=None):
        index = self._get_dict_index(index)
        return self.reduced_mesh_reduced_dofs_list[index]
        
    def get_auxiliary_reduced_function_space(self, auxiliary_problem, component, index=None):
        assert isinstance(component, tuple)
        assert len(component) > 0
        index = self._get_dict_index(index)
        auxiliary_V = _sub_from_tuple(auxiliary_problem.V, component)
        key = (auxiliary_problem, component, index)
        if not key in self._auxiliary_reduced_function_space:
            auxiliary_reduced_V = self.backend.wrapping.convert_functionspace_to_submesh(auxiliary_V, self.reduced_mesh[index], self._get_auxiliary_reduced_function_space_type(auxiliary_V))
            self._auxiliary_reduced_function_space[key] = auxiliary_reduced_V
            if not self._load_auxiliary_reduced_function_space(key):
                # Get the map between DOFs on auxiliary_V and auxiliary_reduced_V
                (auxiliary_dofs_to_reduced_dofs, _) = self.backend.wrapping.map_functionspaces_between_mesh_and_submesh(auxiliary_V, self.mesh, auxiliary_reduced_V, self.reduced_mesh[index])
                log(DEBUG, "Auxiliary DOFs to reduced DOFs is " + str(auxiliary_dofs_to_reduced_dofs))
                self._auxiliary_dofs_to_reduced_dofs[key] = auxiliary_dofs_to_reduced_dofs
                # Save to file
                self._save_auxiliary_reduced_function_space(key)
        else:
            assert key in self._auxiliary_dofs_to_reduced_dofs
        return self._auxiliary_reduced_function_space[key]
    
    def _get_auxiliary_reduced_function_space_type(self, auxiliary_V):
        if hasattr(auxiliary_V, "_component_to_index"):
            def CustomFunctionSpace(mesh, element):
                return FunctionSpace(mesh, element, components=auxiliary_V._component_to_index)
            return CustomFunctionSpace
        else:
            return FunctionSpace
    
    def get_auxiliary_basis_functions_matrix(self, auxiliary_problem, auxiliary_reduced_problem, component, index=None):
        assert isinstance(component, tuple)
        assert len(component) > 0
        index = self._get_dict_index(index)
        key = (auxiliary_problem, component, index) # the mapping between problem and reduced problem is one to one, so there is no need to store both of them in the key
        if not key in self._auxiliary_basis_functions_matrix:
            auxiliary_reduced_V = self.get_auxiliary_reduced_function_space(auxiliary_problem, component, index)
            if not self._load_auxiliary_basis_functions_matrix(key, auxiliary_reduced_problem, auxiliary_reduced_V):
                self._auxiliary_basis_functions_matrix[key] = self._init_auxiliary_basis_functions_matrix(key, auxiliary_reduced_problem, auxiliary_reduced_V)
                self.backend.wrapping.evaluate_basis_functions_matrix_at_dofs(
                    auxiliary_reduced_problem.Z, self._auxiliary_dofs_to_reduced_dofs[key].keys(), 
                    self._auxiliary_basis_functions_matrix[key], self._auxiliary_dofs_to_reduced_dofs[key].values()
                )
                # Save to file
                self._save_auxiliary_basis_functions_matrix(key)
        return self._auxiliary_basis_functions_matrix[key]
        
    def get_auxiliary_function_interpolator(self, auxiliary_problem, component, index=None):
        assert isinstance(component, tuple)
        assert len(component) > 0
        index = self._get_dict_index(index)
        key = (auxiliary_problem, component, index)
        if not key in self._auxiliary_function_interpolator:
            auxiliary_reduced_V = self.get_auxiliary_reduced_function_space(auxiliary_problem, component, index)
            self._auxiliary_function_interpolator[key] = lambda fun: self.backend.wrapping.evaluate_sparse_function_at_dofs(
                fun, self._auxiliary_dofs_to_reduced_dofs[key].keys(), 
                auxiliary_reduced_V, self._auxiliary_dofs_to_reduced_dofs[key].values()
            )
        return self._auxiliary_function_interpolator[key]
        
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
        
