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

from dolfin import CellFunction, cells, compile_extension_module, DEBUG, File, FunctionSpace, log, Vertex, vertices
try:
    from cbcpost.utils import create_submesh as create_submesh_cbcpost, restriction_map
except ImportError:
    from dolfin import MPI, mpi_comm_world
    assert MPI.size(mpi_comm_world()) == 1, "cbcpost is required to create a ReducedMesh in parallel"
    from dolfin import SubMesh as create_submesh
    def restriction_map(V, reduced_V):
        raise NotImplementedError("restriction_map without cbcpost not implemented yet.")
else:
    # Implement an extended version of cbcpost create_submesh that also assigns shared_entities(0).
    # This is essential because otherwise mid-dofs on shared facets will be duplicated!
    from numpy import array, uintp
    def create_submesh(mesh, marker, marker_id):
        submesh = create_submesh_cbcpost(mesh, marker, marker_id)
        assert len(submesh.topology().shared_entities(0)) == 0
        mpi_comm = mesh.mpi_comm().tompi4py()
        if mpi_comm.size == 1:
            return submesh # no sharing in serial
        else:
            # Get all local vertices
            local_vertices__global_index = list()
            local_vertices__global_index__to__local_index = dict()
            for vertex in vertices(submesh):
                local_vertices__global_index.append(vertex.global_index())
                local_vertices__global_index__to__local_index[vertex.global_index()] = vertex.index()
            # Gather all vertices from all processors
            gathered__local_vertices__global_index = list() # over processor id
            for r in range(mpi_comm.size):
                gathered__local_vertices__global_index.append( mpi_comm.bcast(local_vertices__global_index, root=r) )
            # Create dict from global vertex index to processors sharing it
            global_vertex_index__to__processors = dict()
            for r in range(mpi_comm.size):
                for global_vertex_index in gathered__local_vertices__global_index[r]:
                    if global_vertex_index not in global_vertex_index__to__processors:
                        global_vertex_index__to__processors[global_vertex_index] = list()
                    global_vertex_index__to__processors[global_vertex_index].append(r)
            # Get shared vertices
            shared_vertices = dict()
            rank = mpi_comm.rank
            for (global_vertex_index, processors) in global_vertex_index__to__processors.iteritems():
                assert len(processors) > 0
                if len(processors) > 1: # shared among more than one processor
                    if rank in processors:
                        other_processors_list = list(processors)
                        other_processors_list.remove(rank)
                        other_processors = array(other_processors_list, dtype=uintp)
                        shared_vertices[ local_vertices__global_index__to__local_index[global_vertex_index] ] = other_processors
            # Populate shared_entities(0): cannot do that in python becase each call to shared_entities
            # returns a temporary (copied from a code commented out in cbcpost)
            cpp_code = """
                void set_shared_entities_0(Mesh & submesh, std::size_t idx, const Array<std::size_t>& other_processes)
                {
                    std::set<unsigned int> set_other_processes;
                    for (std::size_t i(0); i < other_processes.size(); i++)
                        set_other_processes.insert(other_processes[i]);
                    submesh.topology().shared_entities(0)[idx] = set_other_processes;
                }
            """
            set_shared_entities_0 = compile_extension_module(cpp_code).set_shared_entities_0

            for (local_index, other_processors) in shared_vertices.iteritems():
                set_shared_entities_0(submesh, local_index, other_processors)
            log(DEBUG, "Local indices of shared vertices " + str(submesh.topology().shared_entities(0).keys()))
            log(DEBUG, "Global indices of shared vertices " + str([Vertex(submesh, local_index).global_index() for local_index in submesh.topology().shared_entities(0).keys()]))
            return submesh
            
    # Moreover, override distribute_meshdata (used internally by create_submesh_cbcpost) to 
    # make sure to remove useless vertices in case of redistribution
    import cbcpost.utils.submesh
    distribute_meshdata_cbcpost = cbcpost.utils.submesh.distribute_meshdata

    def new_distribute_meshdata(cells, vertices):
        (new_cells, new_vertices) = distribute_meshdata_cbcpost(cells, vertices)
        # Remove useless vertices in case of redistribution
        all_vertices = list()
        for c in new_cells:
            all_vertices.extend(c)
        all_vertices = set(all_vertices)
        for (set_index, vertex_index) in enumerate(all_vertices):
            assert set_index <= vertex_index
            if set_index < vertex_index:
                for c in new_cells:
                    for (i_v, v) in enumerate(c):
                        if v == vertex_index:
                            c[i_v] = set_index
                new_vertices[set_index] = new_vertices[vertex_index]
                del new_vertices[vertex_index]
        return (new_cells, new_vertices)
        
    cbcpost.utils.submesh.distribute_meshdata = new_distribute_meshdata
        
from RBniCS.backends.abstract import ReducedMesh as AbstractReducedMesh
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.io import ExportableList
from RBniCS.utils.mpi import is_io_process
from mpi4py.MPI import MAX, SUM

@Extends(AbstractReducedMesh)
@BackendFor("FEniCS", inputs=(FunctionSpace, ))
class ReducedMesh(AbstractReducedMesh):
    def __init__(self, V, original_reduced_mesh_dofs_list=None, original_reduced_mesh=None, original_reduced_mesh_reduced_dofs_list=None, original_reduced_function_space=None):
        AbstractReducedMesh.__init__(self, V)
        #
        self.mesh = V.mesh()
        self.mpi_comm = self.mesh.mpi_comm().tompi4py()
        self.V = V
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
        self.reduced_mesh_dofs_list = ExportableList("pickle") # list of dofs
        if original_reduced_mesh_dofs_list is not None:
            self.reduced_mesh_dofs_list.extend(original_reduced_mesh_dofs_list)
        # Reduced meshes, for all N
        self.reduced_mesh = list() # list (over N) of Mesh
        if original_reduced_mesh is not None:
            self.reduced_mesh.append(original_reduced_mesh)
        # DOFs list (of the reduced mesh) that need to be added at each N
        self.reduced_mesh_reduced_dofs_list = ExportableList("pickle") # list (over N) of list of dofs
        if original_reduced_mesh_reduced_dofs_list is not None:
            self.reduced_mesh_reduced_dofs_list.append(original_reduced_mesh_reduced_dofs_list)
        # Reduced function spaces, for all N
        self.reduced_function_space = list() # list (over N) of FunctionSpace
        if original_reduced_function_space is not None:
            self.reduced_function_space.append(original_reduced_function_space)
    
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
            return False
        else:
            Nmax = self._load_Nmax(directory, filename)
            for index in range(Nmax):
                mesh_filename = str(directory) + "/" + filename + "_" + str(index) + ".xml"
                reduced_mesh = Mesh(mesh_filename)
                self.reduced_mesh.append(reduced_mesh)
                self.reduced_function_space.append(FunctionSpace(reduced_mesh, self.V.ufl_element()))
            self.reduced_mesh_dofs_list.load(directory, filename + "_dofs")
            self.reduced_mesh_reduced_dofs_list.load(directory, filename + "_reduced_dofs")
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
        self._save_Nmax(directory, filename)
        assert len(self.reduced_mesh) == len(self.reduced_mesh_reduced_dofs_list)
        for (index, reduced_mesh) in enumerate(self.reduced_mesh):
            mesh_filename = str(directory) + "/" + filename + "_" + str(index) + ".xml"
            File(mesh_filename) << reduced_mesh
        self.reduced_mesh_dofs_list.save(directory, filename + "_dofs")
        self.reduced_mesh_reduced_dofs_list.save(directory, filename + "_reduced_dofs")
            
    def _save_Nmax(self, directory, filename):
        assert len(self.reduced_mesh) == len(self.reduced_mesh_reduced_dofs_list)
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self.reduced_mesh)))
        self.mpi_comm.barrier()
                
    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start is None 
        assert key.step is None
        assert key.stop > 0
        key_to_index = key.stop - 1
        return ReducedMesh(self.V, self.reduced_mesh_dofs_list[key], self.reduced_mesh[key_to_index], self.reduced_mesh_reduced_dofs_list[key_to_index], self.reduced_function_space[key_to_index])
                
    def _store_reduced_mesh_and_reduced_dofs(self):
        # As described in cbcpost, FEniCS cannot partition a mesh with less
        # cells than processors. If there are too few cells, just give up.
        num_cells = (self.reduced_mesh_cells_marker.array() == 1).sum()
        num_cells = self.mpi_comm.allreduce(num_cells, op=SUM)
        log(DEBUG, "Number of cells: " + str(num_cells))
        num_processors = self.mpi_comm.size
        if (num_processors > num_cells):
            self.reduced_mesh.append(None)
            self.reduced_function_space.append(None)
            self.reduced_mesh_reduced_dofs_list.append(None)
        else: # usual case
            # Create submesh thanks to cbcpost
            reduced_mesh = create_submesh(self.mesh, self.reduced_mesh_cells_marker, 1)
            self.reduced_mesh.append(reduced_mesh)
            # Return the FunctionSpace V on the reduced mesh
            reduced_V = FunctionSpace(reduced_mesh, self.V.ufl_element())
            self.reduced_function_space.append(reduced_V)
            # Get the map between DOFs on reduced_V and V
            reduced_dofs__to__dofs = restriction_map(self.V, reduced_V)
            # ... invert it ...
            dofs__to__reduced_dofs = dict()
            for (reduced_dof, dof) in reduced_dofs__to__dofs.iteritems():
                assert dof not in dofs__to__reduced_dofs
                dofs__to__reduced_dofs[dof] = reduced_dof
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
            log(DEBUG, "corresponding to DOFs list " + str(self.reduced_mesh_dofs_list._list))
            self.reduced_mesh_reduced_dofs_list.append(reduced_mesh_reduced_dofs_list)
        

    def get_reduced_mesh(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_mesh[index]
    
    def get_reduced_function_space(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_function_space[index]
        
    def get_dofs_list(self, index=None):
        if index is None:
            index = len(self.reduced_mesh_dofs_list)
        
        return self.reduced_mesh_dofs_list[:index]
        
    def get_reduced_dofs_list(self, index=None):
        if index is None:
            index = -1
        
        return self.reduced_mesh_reduced_dofs_list[index]
        
