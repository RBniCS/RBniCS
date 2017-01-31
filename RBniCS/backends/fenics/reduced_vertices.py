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
from numpy.linalg import norm
from dolfin import Cell, Mesh, Point
from RBniCS.backends.abstract import ReducedVertices as AbstractReducedVertices
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.io import ExportableList, Folders

@Extends(AbstractReducedVertices)
@BackendFor("fenics", inputs=(Mesh, ))
class ReducedVertices(AbstractReducedVertices):
    def __init__(self, V, **kwargs):
        AbstractReducedVertices.__init__(self, V)
        self._V = V
        self._mesh = V.mesh()
        self._mpi_comm = self._mesh.mpi_comm().tompi4py()
        
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
        # Additional storage for reduced mesh
        if copy_from is None:
            self._reduced_mesh = ReducedMesh((V, ))
        else:
            self._reduced_mesh = ReducedMesh((V, ), copy_from=copy_from._reduced_mesh, key_as_slice=key_as_slice, key_as_int=key_as_int)
        self._local_dof_to_coordinates = V.tabulate_dof_coordinates().reshape((-1, self._mesh.ufl_cell().topological_dimension()))
        
    @override
    def append(self, vertex_and_component):
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
        # Update reduced mesh
        self._reduced_mesh._init_for_append_if_needed()
        global_dof_min = None
        distance_min = None
        if self._is_local[-1]:
            cell_id = self._bounding_box_tree.compute_first_entity_collision(vertex_as_point)
            cell = Cell(self._mesh, cell_id)
            self._reduced_mesh.reduced_mesh_cells_marker[cell] = 1
            global_dof_ownership_range = self._V.dofmap().ownership_range()
            for local_dof in self._V.dofmap().cell_dofs(cell_id):
                global_dof = self._V.dofmap().local_to_global_index(local_dof)
                if global_dof >= global_dof_ownership_range[0] and global_dof < global_dof_ownership_range[1]:
                    distance = norm(self._local_dof_to_coordinates[local_dof] - vertex)
                    if distance_min is None or distance < distance_min:
                        global_dof_min = global_dof
                        distance_min = distance
        all_global_dof_min = self._mpi_comm.allgather(global_dof_min)
        all_distance_min = self._mpi_comm.allgather(distance_min)
        global_dof_min_over_processors = None
        distance_min_over_processors = None
        for (global_dof, distance) in zip(all_global_dof_min, all_distance_min):
            assert (
                (global_dof is None and distance is None)
                    or
                (global_dof is not None and distance is not None)
            )
            if distance is not None:
                if distance_min_over_processors is None or distance < distance_min_over_processors:
                    global_dof_min_over_processors = global_dof
                    distance_min_over_processors = distance
        assert global_dof_min_over_processors is not None
        assert distance_min_over_processors is not None
        self._reduced_mesh.reduced_mesh_dofs_list.append((global_dof_min_over_processors, ))
        self._reduced_mesh._update()
        
    @override
    def save(self, directory, filename):
        # Get full directory name
        full_directory = Folders.Folder(directory + "/" + filename)
        full_directory.create()
        # Vertex and component lists
        self._vertex_list.save(full_directory, "vertices")
        self._component_list.save(full_directory, "components")
        # Save reduced mesh
        self._reduced_mesh.save(directory, filename)
        # Note that is local cannot be saved, since partitioning may change due to 
        # different number of MPI processes
        
    @override
    def load(self, directory, filename):
        # Get full directory name
        full_directory = directory + "/" + filename
        # Vertex and component lists
        vertex_import_successful = self._vertex_list.load(full_directory, "vertices")
        component_import_successful = self._component_list.load(full_directory, "components")
        assert vertex_import_successful == component_import_successful
        # Load reduced mesh
        self._reduced_mesh.load(directory, filename)
        # Recompute is_local map, if required
        if len(self._is_local) == 0: # avoid computing multiple times
            for vertex in self._vertex_list:
                vertex_as_point = Point(vertex)
                self._is_local.append(self._bounding_box_tree.collides_entity(vertex_as_point))
        # Return
        return vertex_import_successful and component_import_successful
        
    @override
    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start is None 
        assert key.step is None
        return ReducedVertices(self._V, copy_from=self, key_as_slice=key, key_as_int=key.stop - 1)
                
    def is_local(self, index):
        return self._is_local[index]
        
    def get_reduced_mesh(self, index=None):
        return self._reduced_mesh.get_reduced_mesh(index)
    
    def get_reduced_function_space(self, index=None):
        output = self._reduced_mesh.get_reduced_function_spaces(index)
        assert isinstance(output, tuple)
        assert len(output) == 1
        return output[0]
        
    def get_reduced_subdomain_data(self, index=None):
        return self._reduced_mesh.get_reduced_subdomain_data(index)
        
    def get_dofs_list(self, index=None):
        return self._reduced_mesh.get_dofs_list(index)
        
    def get_reduced_dofs_list(self, index=None):
        return self._reduced_mesh.get_reduced_dofs_list(index)
        
    def get_auxiliary_reduced_function_space(self, auxiliary_problem, index=None):
        return self._reduced_mesh.get_auxiliary_reduced_function_space(auxiliary_problem, index)
                
    def get_auxiliary_basis_functions_matrix(self, auxiliary_problem, auxiliary_reduced_problem, index=None):
        return self._reduced_mesh.get_auxiliary_basis_functions_matrix(auxiliary_problem, auxiliary_reduced_problem, index)
        
