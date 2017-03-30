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
from dolfin import as_backend_type, Function, FunctionSpace, SLEPcEigenSolver
from rbnics.backends.fenics.affine_expansion_storage import AffineExpansionStorage
from rbnics.backends.fenics.matrix import Matrix
from rbnics.backends.abstract import EigenSolver as AbstractEigenSolver
from rbnics.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractEigenSolver)
@BackendFor("fenics", inputs=(FunctionSpace, Matrix.Type(), (Matrix.Type(), None), (AffineExpansionStorage, None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, V, A, B=None, bcs=None):
        self.V = V
        self.A = A
        self.B = B
        self.bcs = bcs
        if self.bcs is not None:
            assert self.bcs.type() == "DirichletBC"
            # Create a copy of A and B in order not to change the original references 
            # when applying bcs by clearing constrained dofs
            self.A = self.A.copy()
            if self.B is not None:
                self.B = self.B.copy()
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

    def _clear_constrained_dofs(self, operator, diag_value):
        for bc_list in self.bcs._content:
            for bc in bc_list:
                constrained_dofs = [bc.function_space.dofmap().local_to_global_index(local_dof_index) for local_dof_index in bc.get_boundary_values().keys()]
                as_backend_type(operator).mat().zeroRowsColumns(constrained_dofs, diag_value)
        
    @override
    def set_parameters(self, parameters, skip_init=False):
        self._parameters = parameters
        assert "spectrum" in parameters
        assert parameters["spectrum"] in ("largest real", "smallest real")
        self._spectrum = parameters["spectrum"]
        if not skip_init and "spectral_shift" in parameters:
            self._init_eigensolver(1./abs(parameters["spectral_shift"])) # we will check in solve if it is an appropriate spurious eigenvalue
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
                    if self._spurious_eigenvalue - smallest_computed_eigenvalue >= 0. or isclose(self._spurious_eigenvalue - smallest_computed_eigenvalue, 0.):
                        new_spurious_eigenvalue = 0.1*smallest_computed_eigenvalue
                        if new_spurious_eigenvalue == self._spurious_eigenvalue: # recursion limit was exhausted
                            return False
                        self._init_eigensolver(new_spurious_eigenvalue)
                        self.set_parameters(self._parameters, skip_init=True)
                        return True
                    else:
                        return False
                elif self._spectrum == "smallest real":
                    largest_computed_eigenvalue, largest_computed_eigenvalue_imag = self.eigen_solver.get_eigenvalue(n_eigs - 1)
                    assert isclose(largest_computed_eigenvalue_imag, 0), "The required eigenvalue is not real"
                    if self._spurious_eigenvalue - largest_computed_eigenvalue <= 0. or isclose(self._spurious_eigenvalue - largest_computed_eigenvalue, 0.):
                        new_spurious_eigenvalue = 10.*largest_computed_eigenvalue
                        if new_spurious_eigenvalue == self._spurious_eigenvalue: # recursion limit was exhausted
                            return False
                        self._init_eigensolver(new_spurious_eigenvalue)
                        self.set_parameters(self._parameters, skip_init=True)
                        return True
                    else:
                        return False
            else:
                return False
                
        do_solve()
        while have_spurious_eigenvalue():
            do_solve() # the spurious eigenvalue has been changed by the previous call to have_spurious_eigenvalue
    
    @override
    def get_eigenvalue(self, i):
        return self.eigen_solver.get_eigenvalue(i)
    
    @override
    def get_eigenvector(self, i):
        (_, _, real_vector, imag_vector) = self.eigen_solver.get_eigenpair(i)
        return (Function(self.V, real_vector), Function(self.V, imag_vector))
        
