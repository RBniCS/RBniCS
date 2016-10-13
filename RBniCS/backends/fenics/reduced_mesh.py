# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import CellFunction, cells, DEBUG, File, FunctionSpace, has_hdf5, log, Mesh, MeshFunction
if has_hdf5():
    from dolfin import HDF5File        
from RBniCS.backends.abstract import ReducedMesh as AbstractReducedMesh
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.io import ExportableList
from RBniCS.utils.mpi import is_io_process
from mpi4py.MPI import MAX, SUM
from RBniCS.backends.fenics.wrapping_utils import build_dof_map_reader_mapping, build_dof_map_writer_mapping, create_submesh, create_submesh_subdomains, mesh_dofs_to_submesh_dofs

@Extends(AbstractReducedMesh)
@BackendFor("FEniCS", inputs=(FunctionSpace, ))
class ReducedMesh(AbstractReducedMesh):
    def __init__(self, V, subdomain_data=None, original_reduced_mesh_dofs_list=None, original_reduced_mesh=None, original_reduced_subdomain_data=None, original_reduced_mesh_reduced_dofs_list=None, original_reduced_function_space=None):
        AbstractReducedMesh.__init__(self, V)
        #
        self.mesh = V.mesh()
        self.mpi_comm = self.mesh.mpi_comm().tompi4py()
        self.V = V
        self.subdomain_data = subdomain_data
        # Store an auxiliary dof to cell dict
        self.dof_to_cells = dict()
        for cell in cells(self.mesh):
            local_dofs = V.dofmap().cell_dofs(cell.index())
            for local_dof in local_dofs:
                global_dof = V.dofmap().local_to_global_index(local_dof)
                if not global_dof in self.dof_to_cells:
                    self.dof_to_cells[global_dof] = list()
                if not cell in self.dof_to_cells[global_dof]:
                    self.dof_to_cells[global_dof].append(cell)
        # Debugging
        log(DEBUG, "DOFs to cells map on processor " + str(self.mpi_comm.rank) + ":")
        for (global_dof, cells_) in self.dof_to_cells.iteritems():
            log(DEBUG, "\t" + str(global_dof) + ": " + str([cell.global_index() for cell in cells_]))
        # Cell function to mark cells (on the full mesh)
        self.reduced_mesh_cells_marker = CellFunction("size_t", self.mesh, 0)
        # DOFs list (of the full mesh) that need to be added at each N
        self.reduced_mesh_dofs_list = list() # list of dofs
        if original_reduced_mesh_dofs_list is not None:
            self.reduced_mesh_dofs_list.extend(original_reduced_mesh_dofs_list)
            # Dot not waste time recreating mappings which are used only for I/O
            self.reduced_mesh_dofs_list__dof_map_writer_mapping = None
            self.reduced_mesh_dofs_list__dof_map_reader_mapping = None
        else:
            self.reduced_mesh_dofs_list__dof_map_writer_mapping = build_dof_map_writer_mapping(V)
            self.reduced_mesh_dofs_list__dof_map_reader_mapping = build_dof_map_reader_mapping(V)
        # Reduced meshes, for all N
        self.reduced_mesh = list() # list (over N) of Mesh
        if original_reduced_mesh is not None:
            self.reduced_mesh.append(original_reduced_mesh)
        # Reduced subdomain data, for all N
        self.reduced_subdomain_data = list() # list (over N) of dict from mesh MeshFunction to reduced_mesh MeshFunction
        if original_reduced_subdomain_data is not None:
            self.reduced_subdomain_data.append(original_reduced_subdomain_data)
        # Reduced function spaces, for all N
        self.reduced_function_space = list() # list (over N) of FunctionSpace
        if original_reduced_function_space is not None:
            self.reduced_function_space.append(original_reduced_function_space)
        # DOFs list (of the reduced mesh) that need to be added at each N
        self.reduced_mesh_reduced_dofs_list = list() # list (over N) of list of dofs
        if original_reduced_mesh_reduced_dofs_list is not None:
            self.reduced_mesh_reduced_dofs_list.append(original_reduced_mesh_reduced_dofs_list)
            # Dot not waste time recreating mappings which are used only for I/O
            self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping = None
            self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping = None
        else:
            self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping = list() # over N
            self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping = list() # over N
        
    @override
    def append(self, global_dofs):
        # Initialize it only the first time (it is not initialized in the constructor to avoid wasting time online)
        if self.reduced_mesh_cells_marker is None:
            self.reduced_mesh_cells_marker = CellFunction("size_t", self.mesh, 0)
        # 
        assert isinstance(global_dofs, tuple)
        assert len(global_dofs) in (1, 2)
        self.reduced_mesh_dofs_list.append(global_dofs)
        # Mark all cells (with an increasing marker)
        for global_dof in global_dofs:
            global_dof_found = 0
            if global_dof in self.dof_to_cells:
                global_dof_found = 1
                for cell in self.dof_to_cells[global_dof]:
                    self.reduced_mesh_cells_marker[cell] = 1
            global_dof_found = self.mpi_comm.allreduce(global_dof_found, op=MAX)
            assert global_dof_found == 1
        # Process marked cells
        self._store_reduced_mesh_and_reduced_dofs()
    
    @override
    def load(self, directory, filename):
        if len(self.reduced_mesh) > 0: # avoid loading multiple times
            self._assert_list_lengths()
            return False
        else:
            self._assert_list_lengths()
            # Nmax
            Nmax = self._load_Nmax(directory, filename)
            # reduced_mesh
            for index in range(Nmax):
                mesh_filename = str(directory) + "/" + filename + "_" + str(index)
                if not has_hdf5():
                    assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
                    mesh_filename = mesh_filename + ".xml"
                    reduced_mesh = Mesh(mesh_filename)
                else:
                    mesh_filename = mesh_filename + ".h5"
                    reduced_mesh = Mesh()
                    input_file = HDF5File(self.mesh.mpi_comm(), mesh_filename, "r")
                    input_file.read(reduced_mesh, "/mesh", False)
                    input_file.close()
                self.reduced_mesh.append(reduced_mesh)
                self.reduced_function_space.append(FunctionSpace(reduced_mesh, self.V.ufl_element()))
            # reduced_subdomain_data
            for index in range(Nmax):
                if self.subdomain_data is not None:
                    reduced_subdomain_data = dict()
                    for (subdomain_index, subdomain) in enumerate(self.subdomain_data):
                        subdomain_filename = str(directory) + "/" + filename + "_" + str(index) + "_subdomain_" + str(subdomain_index)
                        if not has_hdf5():
                            assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
                            subdomain_filename = subdomain_filename + ".xml"
                            reduced_subdomain = MeshFunction("size_t", self.reduced_mesh[index], subdomain_filename)
                        else:
                            subdomain_filename = subdomain_filename + ".h5"
                            input_file = HDF5File(self.mesh.mpi_comm(), subdomain_filename, "r")
                            reduced_subdomain = MeshFunction("size_t", self.reduced_mesh[index], subdomain.dim())
                            input_file.read(reduced_subdomain, "/subdomain")
                            input_file.close()
                        reduced_subdomain_data[subdomain] = reduced_subdomain
                    self.reduced_subdomain_data.append(reduced_subdomain_data)
                else:
                    self.reduced_subdomain_data.append(0) # cannot use None because otherwise it would not be appended by the copy constructor
            # reduced_mesh_dofs_list
            importable_reduced_mesh_dofs_list = ExportableList("pickle")
            importable_reduced_mesh_dofs_list.load(directory, filename + "_dofs")
            assert len(self.reduced_mesh_dofs_list) == 0
            importable_reduced_mesh_dofs_list__iterator = 0
            while importable_reduced_mesh_dofs_list__iterator < len(importable_reduced_mesh_dofs_list):
                if importable_reduced_mesh_dofs_list__iterator == 0:
                    importable_reduced_mesh_dofs_list_tuple_length = importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator]
                    importable_reduced_mesh_dofs_list__iterator += 1
                else:
                    reduced_mesh_dof = list()
                    for i in range(importable_reduced_mesh_dofs_list_tuple_length):
                        (global_cell_index, cell_dof) = (importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator][0], importable_reduced_mesh_dofs_list[importable_reduced_mesh_dofs_list__iterator][1])
                        reduced_mesh_dof.append( self.reduced_mesh_dofs_list__dof_map_reader_mapping[global_cell_index][cell_dof] )
                        importable_reduced_mesh_dofs_list__iterator += 1
                    self.reduced_mesh_dofs_list.append(tuple(reduced_mesh_dof))
            # reduced_mesh_reduced_dofs_list
            for index in range(Nmax):
                assert len(self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping) == index
                self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping.append( build_dof_map_reader_mapping(self.reduced_function_space[index]) )
                importable_reduced_mesh_reduced_dofs_list = ExportableList("pickle")
                importable_reduced_mesh_reduced_dofs_list.load(directory, filename + "_reduced_dofs_" + str(index))
                assert len(self.reduced_mesh_reduced_dofs_list) == index
                self.reduced_mesh_reduced_dofs_list.append( list() )
                importable_reduced_mesh_reduced_dofs_list__iterator = 0
                while importable_reduced_mesh_reduced_dofs_list__iterator < len(importable_reduced_mesh_reduced_dofs_list):
                    if importable_reduced_mesh_reduced_dofs_list__iterator == 0:
                        importable_reduced_mesh_reduced_dofs_list_tuple_length = importable_reduced_mesh_reduced_dofs_list[importable_reduced_mesh_reduced_dofs_list__iterator]
                        importable_reduced_mesh_reduced_dofs_list__iterator += 1
                    else:
                        reduced_mesh_dof = list()
                        for i in range(importable_reduced_mesh_reduced_dofs_list_tuple_length):
                            (global_cell_index, cell_dof) = (importable_reduced_mesh_reduced_dofs_list[importable_reduced_mesh_reduced_dofs_list__iterator][0], importable_reduced_mesh_reduced_dofs_list[importable_reduced_mesh_reduced_dofs_list__iterator][1])
                            reduced_mesh_dof.append( self.reduced_mesh_reduced_dofs_list__dof_map_reader_mapping[index][global_cell_index][cell_dof] )
                            importable_reduced_mesh_reduced_dofs_list__iterator += 1
                        self.reduced_mesh_reduced_dofs_list[index].append(tuple(reduced_mesh_dof))
            #
            self._assert_list_lengths()
            return True
        
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self.mpi_comm.bcast(Nmax, root=is_io_process.root)
        return Nmax
        
    @override
    def save(self, directory, filename):
        self._assert_list_lengths()
        # Nmax
        self._save_Nmax(directory, filename)
        # reduced_mesh
        for (index, reduced_mesh) in enumerate(self.reduced_mesh):
            mesh_filename = str(directory) + "/" + filename + "_" + str(index)
            if not has_hdf5():
                assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
                mesh_filename = mesh_filename + ".xml"
                File(mesh_filename) << reduced_mesh
            else:
                mesh_filename = mesh_filename + ".h5"
                output_file = HDF5File(self.mesh.mpi_comm(), mesh_filename, "w")
                output_file.write(reduced_mesh, "/mesh")
                output_file.close()
        # reduced_subdomain_data
        if self.subdomain_data is not None:
            for (index, reduced_subdomain_data) in enumerate(self.reduced_subdomain_data):
                subdomain_index = 0
                for (subdomain, reduced_subdomain) in reduced_subdomain_data.iteritems():
                    subdomain_filename = str(directory) + "/" + filename + "_" + str(index) + "_subdomain_" + str(subdomain_index)
                    if not has_hdf5():
                        assert self.mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh function in parallel"
                        subdomain_filename = subdomain_filename + ".xml"
                        File(subdomain_filename) << reduced_subdomain
                    else:
                        subdomain_filename = subdomain_filename + ".h5"
                        output_file = HDF5File(self.mesh.mpi_comm(), subdomain_filename, "w")
                        output_file.write(reduced_subdomain, "/subdomain")
                        output_file.close()
                    subdomain_index += 1
        # reduced_mesh_dofs_list
        exportable_reduced_mesh_dofs_list = ExportableList("pickle")
        exportable_reduced_mesh_dofs_list.append(len(self.reduced_mesh_dofs_list[0])) # it is the same for all dofs
        for reduced_mesh_dof in self.reduced_mesh_dofs_list:
            for reduced_mesh_dof__component in reduced_mesh_dof:
                exportable_reduced_mesh_dofs_list.append(self.reduced_mesh_dofs_list__dof_map_writer_mapping[reduced_mesh_dof__component])
        exportable_reduced_mesh_dofs_list.save(directory, filename + "_dofs")
        # reduced_mesh_reduced_dofs_list
        assert len(self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping) == len(self.reduced_mesh) - 1
        self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping.append( build_dof_map_writer_mapping( self.reduced_function_space[-1] ) )
        for (index, reduced_mesh_reduced_dofs_list) in enumerate(self.reduced_mesh_reduced_dofs_list):
            exportable_reduced_mesh_reduced_dofs_list = ExportableList("pickle")
            exportable_reduced_mesh_reduced_dofs_list.append(len(reduced_mesh_reduced_dofs_list[0])) # it is the same for all dofs
            for reduced_mesh_reduced_dof in reduced_mesh_reduced_dofs_list:
                for reduced_mesh_reduced_dof__component in reduced_mesh_reduced_dof:
                    exportable_reduced_mesh_reduced_dofs_list.append(self.reduced_mesh_reduced_dofs_list__dof_map_writer_mapping[index][reduced_mesh_reduced_dof__component])
            exportable_reduced_mesh_reduced_dofs_list.save(directory, filename + "_reduced_dofs_" + str(index))
            
    def _save_Nmax(self, directory, filename):
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self.reduced_mesh)))
        self.mpi_comm.barrier()
        
    def _assert_list_lengths(self):
        assert len(self.reduced_mesh) == len(self.reduced_function_space)
        assert len(self.reduced_mesh) == len(self.reduced_subdomain_data)
        assert len(self.reduced_mesh) == len(self.reduced_mesh_dofs_list)
        assert len(self.reduced_mesh) == len(self.reduced_mesh_reduced_dofs_list)
                
    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start is None 
        assert key.step is None
        assert key.stop > 0
        key_to_index = key.stop - 1
        return ReducedMesh(self.V, self.subdomain_data, self.reduced_mesh_dofs_list[key], self.reduced_mesh[key_to_index], self.reduced_subdomain_data[key_to_index], self.reduced_mesh_reduced_dofs_list[key_to_index], self.reduced_function_space[key_to_index])
                
    def _store_reduced_mesh_and_reduced_dofs(self):
        # Create submesh
        reduced_mesh = create_submesh(self.mesh, self.reduced_mesh_cells_marker, 1)
        self.reduced_mesh.append(reduced_mesh)
        # Create subdomain data on submesh
        if self.subdomain_data is not None:
            reduced_subdomain_data_list = create_submesh_subdomains(self.mesh, reduced_mesh, self.subdomain_data)
            reduced_subdomain_data = dict()
            assert len(self.subdomain_data) == len(reduced_subdomain_data_list)
            for (subdomain, reduced_subdomain) in zip(self.subdomain_data, reduced_subdomain_data_list):
                reduced_subdomain_data[subdomain] = reduced_subdomain
            self.reduced_subdomain_data.append(reduced_subdomain_data)
        else:
            self.reduced_subdomain_data.append(0) # cannot use None because otherwise it would not be appended by the copy constructor
        # Return the FunctionSpace V on the reduced mesh
        reduced_V = FunctionSpace(reduced_mesh, self.V.ufl_element())
        self.reduced_function_space.append(reduced_V)
        # Get the map between DOFs on V and reduced_V
        dofs__to__reduced_dofs = mesh_dofs_to_submesh_dofs(self.V, reduced_V)
        log(DEBUG, "DOFs to reduced DOFs is " + str(dofs__to__reduced_dofs))
        # ... and fill in reduced_mesh_reduced_dofs_list ...
        reduced_mesh_reduced_dofs_list = list()
        for dofs in self.reduced_mesh_dofs_list:
            reduced_dofs = list()
            for dof in dofs:
                dof_processor = -1
                reduced_dof = None
                if dof in dofs__to__reduced_dofs:
                    reduced_dof = dofs__to__reduced_dofs[dof]
                    dof_processor = self.mpi_comm.rank
                dof_processor = self.mpi_comm.allreduce(dof_processor, op=MAX)
                assert dof_processor >= 0
                reduced_dofs.append(self.mpi_comm.bcast(reduced_dof, root=dof_processor))
            assert len(reduced_dofs) in (1, 2)
            reduced_mesh_reduced_dofs_list.append(tuple(reduced_dofs))
        log(DEBUG, "Reduced DOFs list " + str(reduced_mesh_reduced_dofs_list))
        log(DEBUG, "corresponding to DOFs list " + str(self.reduced_mesh_dofs_list))
        self.reduced_mesh_reduced_dofs_list.append(reduced_mesh_reduced_dofs_list)

    def get_reduced_mesh(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_mesh[index]
    
    def get_reduced_function_space(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_function_space[index]
        
    def get_reduced_subdomain_data(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_subdomain_data[index]
        
    def get_dofs_list(self, index=None):
        if index is None:
            index = len(self.reduced_mesh_dofs_list)
        
        return self.reduced_mesh_dofs_list[:index]
        
    def get_reduced_dofs_list(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_mesh_reduced_dofs_list[index]
        
