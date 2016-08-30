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

from dolfin import as_backend_type, Function, FunctionSpace, SLEPcEigenSolver
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.abstract import EigenSolver as AbstractEigenSolver
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractEigenSolver)
@BackendFor("FEniCS", inputs=(Matrix.Type(), (Matrix.Type(), None), (FunctionSpace, None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, A, B=None, V=None):
        A = as_backend_type(A)
        if B is not None:
            B = as_backend_type(B)
        self.eigen_solver = SLEPcEigenSolver(A, B)
        assert V is not None
        self.V = V
        
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
        (_, _, real_vector, imag_vector) = self.eigen_solver.get_eigenpair(i)
        return (Function(self.V, real_vector), Function(self.V, imag_vector))
        
