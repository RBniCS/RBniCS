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

from dolfin import CellFunction, cells, File, FunctionSpace
try:
    from cbcpost.utils import create_submesh, restriction_map
except ImportError:
    from dolfin import MPI, mpi_comm_world
    assert MPI.size(mpi_comm_world()) == 1, "cbcpost is required to create a ReducedMesh in parallel"
    from dolfin import SubMesh as create_submesh
    def restriction_map(V, reduced_V):
        raise NotImplementedError("restriction_map without cbcpost not implemented yet.")
from RBniCS.backends.abstract import ReducedMesh as AbstractReducedMesh
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.io import ExportableList

@Extends(AbstractReducedMesh)
@BackendFor("FEniCS", inputs=(FunctionSpace, ))
class ReducedMesh(AbstractReducedMesh):
    def __init__(self, V):
        AbstractReducedMesh.__init__(self, V)
        #
        self.mesh = V.mesh()
        self.V = V
        # Store an auxiliary dof to cell dict
        self.dof_to_cells = dict()
        for cell in cells(self.mesh):
            dofs = V.dofmap().cell_dofs(cell.index())
            for dof in dofs:
                if not dof in self.dof_to_cells:
                    self.dof_to_cells[dof] = list()
                if not cell.index() in self.dof_to_cells[dof]:
                    self.dof_to_cells[dof].append(cell.index())
        # Cell function to mark cells (on the full mesh)
        self.reduced_mesh_cells_marker = CellFunction("size_t", self.mesh, 0)
        # DOFs list (of the full mesh) that need to be added at each N
        self.reduced_mesh_dofs_list = list() # list of dofs
        # Reduced meshes, for all N
        self.reduced_mesh = list() # of Mesh
        # DOFs list (of the reduced mesh) that need to be added at each N
        self.reduced_mesh_reduced_dofs_list = ExportableList("pickle") # list (over N) of list of dofs
    
    @override
    def add_dofs(self, dofs):
        # Initialize it only the first time (it is not initialized in the constructor to avoid wasting time online)
        if self.reduced_mesh_cells_marker is None:
            self.reduced_mesh_cells_marker = CellFunction("size_t", self.mesh, 0)
        # 
        assert isinstance(dofs, tuple)
        assert len(dofs) in (1, 2)
        assert dofs not in self.reduced_mesh_dofs_list, "EIM algorithm ensures that at least one new dof is added every time"
        self.reduced_mesh_dofs_list.append(dofs)
        # Mark all cells (with an increasing marker)
        for dof in dofs:
            for cell in self.dof_to_cells[dof]:
                self.reduced_mesh_cells_marker[cell] = 1
        # Process marked cells
        self._store_reduced_mesh_and_reduced_dofs()
    
    @override
    def load(self, directory, filename):
        if len(self._list) > 0: # avoid loading multiple times
            return False
        else:
            Nmax = self._load_Nmax(directory, filename)
            for index in range(Nmax):
                mesh_filename = str(directory) + "/" + filename + "_" + str(index) + ".xml"
                reduced_mesh = Mesh(mesh_filename)
                self.reduced_mesh.append(reduced_mesh)
                reduced_mesh_reduced_dofs_list.load(directory, filename + "_" + str(index) + "_dofs.xml")
                self.reduced_mesh_reduced_dofs_list.append(reduced_mesh_reduced_dofs_list)
            return True
        
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self.mesh.mpi_comm()):
            with open(str(directory) + "/" + filename + ".length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self.mesh.mpi_comm().bcast(Nmax, root=is_io_process.root)
        return Nmax
        
    @override
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        assert len(self.reduced_mesh) == len(self.reduced_mesh_reduced_dofs_list)
        for (index, (reduced_mesh, reduced_mesh_reduced_dofs_list)) in enumerate(zip(self.reduced_mesh, self.reduced_mesh_reduced_dofs_list)):
            mesh_filename = str(directory) + "/" + filename + "_" + str(index) + ".xml"
            File(mesh_filename) << mesh
            reduced_mesh_reduced_dofs_list.save(directory, filename + "_" + str(index) + "_dofs.xml")
            
    def _save_Nmax(self, directory, filename):
        assert len(self.reduced_mesh) == len(self.reduced_mesh_reduced_dofs_list)
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self.reduced_mesh)))
        self.mesh.mpi_comm().barrier()
                
    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start is None 
        assert key.step is None
        return self._create_reduced_function_space(self.reduced_mesh[key.stop]), self.reduced_mesh_reduced_dofs_list[key.stop]
                
    def _store_reduced_mesh_and_reduced_dofs(self):
        # Create submesh thanks to cbcpost
        reduced_mesh = create_submesh(self.mesh, self.reduced_mesh_cells_marker, 1)
        # Return the FunctionSpace V on the reduced mesh
        reduced_V = self._create_reduced_function_space(reduced_mesh)
        # Get the map between DOFs on reduced_V and V
        reduced_dofs__to__dofs = restriction_map(self.V, reduced_V)
        # ... invert it ...
        dofs__to__reduced_dofs = dict()
        for (reduced_dof, dof) in reduced_dofs__to__dofs.iteritems():
            dofs__to__reduced_dofs[dof] = reduced_dof
        # ... and fill in reduced_mesh_reduced_dofs_list ...
        reduced_mesh_reduced_dofs_list = list()
        for dofs in self.reduced_mesh_dofs_list:
            reduced_dofs = list()
            for dof in dofs:
                reduced_dofs.append(dofs__to__reduced_dofs[dof])
            reduced_mesh_reduced_dofs_list.append(tuple(reduced_dofs))
        # Add to storage
        self.reduced_mesh.append(reduced_mesh)
        self.reduced_mesh_reduced_dofs_list.append(reduced_mesh_reduced_dofs_list)
    
    def _create_reduced_function_space(self, reduced_mesh):
        return FunctionSpace(reduced_mesh, self.V.ufl_element())
        
