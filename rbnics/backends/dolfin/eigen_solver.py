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

from numpy import isclose
from petsc4py import PETSc
from ufl import Form
from dolfin import as_backend_type, assemble, DirichletBC, Function, FunctionSpace, PETScMatrix, PETScVector, SLEPcEigenSolver
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.backends.abstract import EigenSolver as AbstractEigenSolver
from rbnics.utils.decorators import BackendFor, dict_of, Extends, list_of, override

@Extends(AbstractEigenSolver)
@BackendFor("dolfin", inputs=(FunctionSpace, (Matrix.Type(), Form), (Matrix.Type(), Form, None), (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, V, A, B=None, bcs=None):
        self.V = V
        if bcs is not None:
            self._set_boundary_conditions(bcs)
        assert isinstance(A, (Matrix.Type(), Form))
        if isinstance(A, Form):
            A = assemble(A, keep_diagonal=True)
        assert isinstance(B, (Matrix.Type(), Form)) or B is None
        if isinstance(B, Form):
            B = assemble(B, keep_diagonal=True)
        self._set_operators(A, B)
        if self.B is not None:
            self.eigen_solver = SLEPcEigenSolver(self.condensed_A, self.condensed_B)
        else:
            self.eigen_solver = SLEPcEigenSolver(self.condensed_A)
    
    def _set_boundary_conditions(self, bcs):
        # List all local and constrained local dofs
        local_dofs = set()
        constrained_local_dofs = set()
        for bc in bcs:
            dofmap = bc.function_space().dofmap()
            local_range = dofmap.ownership_range()
            local_dofs.update(range(local_range[0], local_range[1]))
            constrained_local_dofs.update([
                dofmap.local_to_global_index(local_dof_index) for local_dof_index in bc.get_boundary_values().keys()
            ])
            
        # List all unconstrained dofs
        unconstrained_local_dofs = local_dofs.difference(constrained_local_dofs)
        unconstrained_local_dofs = list(unconstrained_local_dofs)
        
        # Generate IS accordingly
        comm = bcs[0].function_space().mesh().mpi_comm()
        for bc in bcs:
            assert comm == bc.function_space().mesh().mpi_comm()
        self._is = PETSc.IS().createGeneral(unconstrained_local_dofs, comm)
    
    def _set_operators(self, A, B):
        if hasattr(self, "_is"): # there were Dirichlet BCs
            (self.A, self.condensed_A) = self._condense_matrix(A)
            if B is not None:
                (self.B, self.condensed_B) = self._condense_matrix(B)
            else:
                (self.B, self.condensed_B) = (None, None)
        else:
            (self.A, self.condensed_A) = (as_backend_type(A), as_backend_type(A))
            if B is not None:
                (self.B, self.condensed_B) = (as_backend_type(B), as_backend_type(B))
            else:
                (self.B, self.condensed_B) = (None, None)
    
    def _condense_matrix(self, mat):
        mat = as_backend_type(mat)
        
        petsc_version = PETSc.Sys().getVersionInfo() 
        if petsc_version["major"] == 3 and petsc_version["minor"] <= 7 and petsc_version["release"] is True:
            condensed_mat = mat.mat().getSubMatrix(self._is, self._is)
        else:
            condensed_mat = mat.mat().createSubMatrix(self._is, self._is)

        return mat, PETScMatrix(condensed_mat)
    
    @override
    def set_parameters(self, parameters):
        self.eigen_solver.parameters.update(parameters)
        
    @override
    def solve(self, n_eigs=None):
        assert n_eigs is not None
        self.eigen_solver.solve(n_eigs)
    
    @override
    def get_eigenvalue(self, i):
        return self.eigen_solver.get_eigenvalue(i)
    
    @override
    def get_eigenvector(self, i):
        # Get number of computed eigenvectors/values
        num_computed_eigenvalues = self.eigen_solver.eps().getConverged()

        if (i < num_computed_eigenvalues):
            # Initialize eigenvectors
            real_vector = PETScVector()
            imag_vector = PETScVector()
            self.A.init_vector(real_vector, 0)
            self.A.init_vector(imag_vector, 0)

            # Condense input vectors
            if hasattr(self, "_is"): # there were Dirichlet BCs
                condensed_real_vector = real_vector.vec().getSubVector(self._is)
                condensed_imag_vector = imag_vector.vec().getSubVector(self._is)
            else:
                condensed_real_vector = real_vector
                condensed_imag_vector = imag_vector

            # Get eigenpairs
            _ = self.eigen_solver.eps().getEigenpair(i, condensed_real_vector, condensed_imag_vector)

            # Restore input vectors
            if hasattr(self, "_is"): # there were Dirichlet BCs
                real_vector.vec().restoreSubVector(self._is, condensed_real_vector)
                imag_vector.vec().restoreSubVector(self._is, condensed_imag_vector)
            
            # Return as Function
            return (Function(self.V, real_vector), Function(self.V, imag_vector))
        else:
            raise RuntimeError("Requested eigenpair has not been computed")
            
