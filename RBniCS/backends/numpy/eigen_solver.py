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

from numpy import real, imag
from scipy.linalg import eig, eigh
from RBniCS.backend.abstract import EigenSolver as AbstractEigenSolver
from RBniCS.backend.numpy.matrix import Matrix_Type
from RBniCS.backend.numpy.function import Function
from RBniCS.utils.decorators import any, BackendFor, Extends, override

@Extends(AbstractEigenSolver)
@BackendFor("NumPy", inputs=(Matrix_Type, any(Matrix_Type, None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, V_or_Z, A, B=None):
        assert A.shape[0] == A.shape[1]
        if B is not None:
            assert B.shape[0] == B.shape[1]
            assert A.shape[0] == B.shape[0]
        
        self.parameters = {}
        self.eigs = None
        self.eigv = None
        
    @abstractmethod
    def set_parameters(self, parameters)
        self.parameters.update(parameters)
        
    @override
    def solve(self):
        if self.parameters["problem_type"] == "hermitian":
            eigs, eigv = eigh(A, B)
        else:
            eigs, eigv = eig(A, B)
            
        if self.parameters["spectrum"] == "largest real":
            idx = eigs.argsort() # sort by increasing value
            idx = idx[::-1] # reverse the order
        elif self.parameters["spectrum"] == "smallest real":
            idx = eigs.argsort() # sort by increasing value
        else:
            return ValueError("Invalid spectrum parameter in EigenSolver")
        
        self.eigs = eigs[idx]
        self.eigv = eigv[:, idx]
    
    @override
    def get_eigenvalue(self, i):
        return real(self.eigs[i]), imag(self.eigs[i])
    
    @override
    def get_eigenvector(self, i):
        eigv_i = self.eigv[:, i]
        eigv_i_real = real(eigv_i)
        eigv_i_imag = imag(eigv_i)
        return (Function(eigv_i_real), Function(eigv_i_complex))
        
