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
## @file truth_eigen_solver.py
#  @brief Type of the truth eigen solver
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import isclose
from dolfin import as_backend_type, Function, FunctionSpace, SLEPcEigenSolver
from RBniCS.backends.fenics.affine_expansion_storage import AffineExpansionStorage
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.abstract import EigenSolver as AbstractEigenSolver
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractEigenSolver)
@BackendFor("FEniCS", inputs=(FunctionSpace, Matrix.Type(), (Matrix.Type(), None), (AffineExpansionStorage, None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, V, A, B=None, bcs=None):
        self.V = V
        self.A = A
        if B is not None:
            self.B = B
        else:
            self.B = None
        self.bcs = bcs
        if self.bcs is not None:
            assert self.bcs.type() == "DirichletBC"
            # Create a copy of A and B in order not to change the original references 
            # when applying bcs by clearing constrained dofs
            self.A = self.A.copy()
            if self.B is not None:
                self.B = self.B.copy()
                
            from petsc4py import PETSc
            A_viewer = PETSc.Viewer().createASCII("Ab.m", comm= PETSc.COMM_WORLD)
            A_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
            A_viewer.view(as_backend_type(self.A).mat())
            A_viewer.popFormat()
            b_viewer = PETSc.Viewer().createASCII("Bb.m", comm= PETSc.COMM_WORLD)
            b_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
            b_viewer.view(as_backend_type(self.B).mat())
            b_viewer.popFormat()
            
        self._init_eigensolver(1.) # we will check in solve if it is an appropriate spurious eigenvalue
        
    def _init_eigensolver(self, spurious_eigenvalue):
        if self.bcs is not None:
            self._spurious_eigenvalue = spurious_eigenvalue
            self._clear_constrained_dofs(self.A, spurious_eigenvalue)
            if self.B is not None:
                self._clear_constrained_dofs(self.B, 1.)
        if self.B is not None:
            self.eigen_solver = SLEPcEigenSolver(as_backend_type(self.A), as_backend_type(self.B))
        else:
            self.eigen_solver = SLEPcEigenSolver(as_backend_type(self.A))
            
        if spurious_eigenvalue > 10:
            from petsc4py import PETSc
            A_viewer = PETSc.Viewer().createASCII("A.m", comm= PETSc.COMM_WORLD)
            A_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
            A_viewer.view(as_backend_type(self.A).mat())
            A_viewer.popFormat()
            b_viewer = PETSc.Viewer().createASCII("B.m", comm= PETSc.COMM_WORLD)
            b_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
            b_viewer.view(as_backend_type(self.B).mat())
            b_viewer.popFormat()

    def _clear_constrained_dofs(self, operator, diag_value):
        dummy = Function(self.V)
        for bc_list in self.bcs._content:
            for bc in bc_list:
                bc.zero(operator)
                bc.zero_columns(operator, dummy.vector(), diag_value)
        
    @override
    def set_parameters(self, parameters):
        assert "spectrum" in parameters
        assert parameters["spectrum"] in ("largest real", "smallest real")
        self._spectrum = parameters["spectrum"]
        if "spectral_shift" in parameters:
            self._init_eigensolver(1./parameters["spectral_shift"]) # we will check in solve if it is an appropriate spurious eigenvalue
        self.eigen_solver.parameters.update(parameters)
        
    @override
    def solve(self, n_eigs=None):
        def do_solve():
            assert n_eigs is not None
            self.eigen_solver.solve(n_eigs)
        
        # Check if the spurious eigenvalue related to BCs is part of the computed eigenvalues.
        # If it is, reinit the eigen problem with a different spurious eigenvalue
        def have_spurious_eigenvalue():
            if self.bcs is not None:
                assert self._spectrum in ("largest real", "smallest real")
                if self._spectrum == "largest real":
                    smallest_computed_eigenvalue, smallest_computed_eigenvalue_imag = self.eigen_solver.get_eigenvalue(n_eigs - 1)
                    assert isclose(smallest_computed_eigenvalue_imag, 0), "The required eigenvalue is not real"
                    if self._spurious_eigenvalue >= smallest_computed_eigenvalue:
                        self._init_eigensolver(0.1*smallest_computed_eigenvalue)
                        return True
                    else:
                        return False
                elif self._spectrum == "smallest real":
                    largest_computed_eigenvalue, largest_computed_eigenvalue_imag = self.eigen_solver.get_eigenvalue(n_eigs - 1)
                    assert isclose(largest_computed_eigenvalue_imag, 0), "The required eigenvalue is not real"
                    if self._spurious_eigenvalue <= largest_computed_eigenvalue:
                        self._init_eigensolver(10.*largest_computed_eigenvalue)
                        return True
                    else:
                        return False
            else:
                return False
                
        do_solve()
        if have_spurious_eigenvalue():
            do_solve() # the spurious eigenvalue has been changed by the previous call to have_spurious_eigenvalue
            assert not have_spurious_eigenvalue()
    
    @override
    def get_eigenvalue(self, i):
        return self.eigen_solver.get_eigenvalue(i)
    
    @override
    def get_eigenvector(self, i):
        (_, _, real_vector, imag_vector) = self.eigen_solver.get_eigenpair(i)
        return (Function(self.V, real_vector), Function(self.V, imag_vector))
        
