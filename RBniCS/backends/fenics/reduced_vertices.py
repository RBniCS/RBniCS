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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import ndarray as array
from dolfin import Cell, CellFunction, File, FunctionSpace, has_hdf5, HDF5File, Mesh, Point
from RBniCS.backends.abstract import ReducedVertices as AbstractReducedVertices
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.io import ExportableList, Folders
from RBniCS.utils.mpi import is_io_process
from RBniCS.backends.fenics.wrapping_utils import create_submesh

@Extends(AbstractReducedVertices)
@BackendFor("fenics", inputs=(Mesh, ))
class ReducedVertices(AbstractReducedVertices):
    def __init__(self, V, **kwargs):
        AbstractReducedVertices.__init__(self, V)
        self._V = V
        self._mesh = V.mesh()
        
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
            
        # Vertex storage
        if copy_from is None:
            self._vertex_list = ExportableList("pickle") # list of vertices
            self._component_list = ExportableList("pickle") # list of function components
        else:
            self._vertex_list = copy_from._vertex_list[key_as_slice]
            self._component_list = copy_from._component_list[key_as_slice]
        # Additional storage to detect local vertices
        self._bounding_box_tree = self._mesh.bounding_box_tree()
        self._mpi_comm = self._mesh.mpi_comm().tompi4py()
        if copy_from is None:
            self._is_local = list() # list of bool
        else:
            self._is_local = copy_from._is_local[key_as_slice]
        # Additional storage for reduced mesh:
        # ... cell function to mark cells (on the full mesh)
        self._reduced_mesh_cells_marker = None # will be of type CellFunction
        # ... which is not initialized here for performance reasons
        # ... reduced meshes, for all N
        self._reduced_mesh = list() # list (over N) of Mesh
        if copy_from is not None:
            self._reduced_mesh.append(copy_from._reduced_mesh[key_as_int])
        # ... reduced function spaces, for all N
        self._reduced_function_space = list() # list (over N) of FunctionSpace
        if copy_from is not None:
            self._reduced_function_space.append(copy_from._reduced_function_space[key_as_int])
            
    def _init_for_offline_if_needed(self):
        # Initialize cells marker only the first time
        if self._reduced_mesh_cells_marker is None:
            self._reduced_mesh_cells_marker = CellFunction("size_t", self._mesh, 0)
        
    @override
    def append(self, vertex_and_component):
        self._init_for_offline_if_needed()
        assert isinstance(vertex_and_component, tuple)
        assert len(vertex_and_component) == 2
        assert isinstance(vertex_and_component[0], array)
        assert isinstance(vertex_and_component[1], int)
        vertex = vertex_and_component[0]
        component = vertex_and_component[1]
        self._vertex_list.append(vertex)
        self._component_list.append(component)
        # Initialize _is_local map
        vertex_as_point = Point(vertex)
        self._is_local.append(self._bounding_box_tree.collides_entity(vertex_as_point))
        # Extract the reduced mesh
        if self._is_local[-1]:
            cell_id = self._bounding_box_tree.compute_first_entity_collision(vertex_as_point)
            cell = Cell(self._mesh, cell_id)
            self._reduced_mesh_cells_marker[cell] = 1
        reduced_mesh = create_submesh(self._mesh, self._reduced_mesh_cells_marker, 1)
        self._reduced_mesh.append(reduced_mesh)
        # Append the FunctionSpace V on the reduced mesh
        self._reduced_function_space.append(FunctionSpace(reduced_mesh, self._V.ufl_element()))
        
    @override
    def save(self, directory, filename):
        # Get full directory name
        full_directory = Folders.Folder(directory + "/" + filename)
        full_directory.create()
        # Vertex and component lists
        self._vertex_list.save(full_directory, "vertices")
        self._component_list.save(full_directory, "components")
        # Reduced mesh
        self._save_Nmax(directory, filename)
        for (index, reduced_mesh) in enumerate(self._reduced_mesh):
            mesh_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index)
            if not has_hdf5():
                assert self._mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
                mesh_filename = mesh_filename + ".xml"
                File(mesh_filename) << reduced_mesh
            else:
                mesh_filename = mesh_filename + ".h5"
                output_file = HDF5File(self._mesh.mpi_comm(), mesh_filename, "w")
                output_file.write(reduced_mesh, "/mesh")
                output_file.close()
        # Note that is local cannot be saved, since partitioning may change due to 
        # different number of MPI processes
        
    def _save_Nmax(self, directory, filename):
        if is_io_process(self._mpi_comm):
            with open(str(directory) + "/" + filename + "/" + "reduced_mesh.length", "w") as length:
                length.write(str(len(self._reduced_mesh)))
        self._mpi_comm.barrier()
        
    @override
    def load(self, directory, filename):
        # Get full directory name
        full_directory = directory + "/" + filename
        # Vertex and component lists
        vertex_import_successful = self._vertex_list.load(full_directory, "vertices")
        component_import_successful = self._component_list.load(full_directory, "components")
        assert vertex_import_successful == component_import_successful
        # Reduced mesh
        if len(self._reduced_mesh) == 0: # avoid loading multiple times
            Nmax = self._load_Nmax(directory, filename)
            for index in range(Nmax):
                mesh_filename = str(directory) + "/" + filename + "/" + "reduced_mesh_" + str(index)
                if not has_hdf5():
                    assert self._mpi_comm.size == 1, "hdf5 is required by dolfin to save a mesh in parallel"
                    mesh_filename = mesh_filename + ".xml"
                    reduced_mesh = Mesh(mesh_filename)
                else:
                    mesh_filename = mesh_filename + ".h5"
                    reduced_mesh = Mesh()
                    input_file = HDF5File(self._mesh.mpi_comm(), mesh_filename, "r")
                    input_file.read(reduced_mesh, "/mesh", False)
                    input_file.close()
                self._reduced_mesh.append(reduced_mesh)
                # Append the FunctionSpace V on the reduced mesh
                self._reduced_function_space.append(FunctionSpace(reduced_mesh, self._V.ufl_element()))
        # Recompute is_local map, if required
        if len(self._is_local) == 0: # avoid computing multiple times
            for vertex in self._vertex_list:
                vertex_as_point = Point(vertex)
                self._is_local.append(self._bounding_box_tree.collides_entity(vertex_as_point))
        # Return
        return vertex_import_successful and component_import_successful
        
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self._mpi_comm):
            with open(str(directory) + "/" + filename + "/" + "reduced_mesh.length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self._mpi_comm.bcast(Nmax, root=is_io_process.root)
        return Nmax
        
    @override
    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start is None 
        assert key.step is None
        return ReducedVertices(self._V, copy_from=self, key_as_slice=key, key_as_int=key.stop - 1)
                
    def is_local(self, index):
        return self._is_local[index]
        
    def get_reduced_mesh(self, index=None):
        if index is None:
            index = -1
        
        return self._reduced_mesh[index]
        
    def get_reduced_function_space(self, index=None):
        if index is None:
            index = -1
        
        return self._reduced_function_space[index]
        
