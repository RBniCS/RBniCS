# Copyright (C) 2015-2019 by the RBniCS authors
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

from numpy import real, imag
from scipy.linalg import eig, eigh
from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.abstract import EigenSolver as AbstractEigenSolver
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import BackendFor, DictOfThetaType, ThetaType

@BackendFor("numpy", inputs=((AbstractFunctionsList, None), Matrix.Type(), (Matrix.Type(), None), ThetaType + DictOfThetaType + (None,)))
class EigenSolver(AbstractEigenSolver):
    def __init__(self, basis_functions, A, B=None, bcs=None):
        assert A.N == A.M
        if B is not None:
            assert B.N == B.M
            assert A.N == B.M
        
        self.A = A
        self.B = B
        self.parameters = dict()
        self.eigs = None
        self.eigv = None
        assert bcs is None # the case bcs != None has not been implemented yet
        
    def set_parameters(self, parameters):
        self.parameters.update(parameters)
        
    def solve(self, n_eigs=None):
        assert "problem_type" in self.parameters
        if self.parameters["problem_type"] in ("hermitian", "gen_hermitian"):
            eigs, eigv = eigh(self.A, self.B)
        else:
            eigs, eigv = eig(self.A, self.B)
            
        assert "spectrum" in self.parameters
        if self.parameters["spectrum"] == "largest real":
            idx = eigs.argsort() # sort by increasing value
            idx = idx[::-1] # reverse the order
        elif self.parameters["spectrum"] == "smallest real":
            idx = eigs.argsort() # sort by increasing value
        else:
            return ValueError("Invalid spectrum parameter in EigenSolver")
        
        if n_eigs is not None:
            idx = idx[:n_eigs]
        
        self.eigs = eigs[idx]
        self.eigv = eigv[:, idx]
    
    def get_eigenvalue(self, i):
        return float(real(self.eigs[i])), float(imag(self.eigs[i]))
    
    def get_eigenvector(self, i):
        eigv_i = self.eigv[:, i]
        eigv_i_real = Vector.Type()(self.A.N, real(eigv_i))
        eigv_i_imag = Vector.Type()(self.A.N, imag(eigv_i))
        eigv_i_real_fun = Function(eigv_i_real)
        eigv_i_imag_fun = Function(eigv_i_imag)
        return (eigv_i_real_fun, eigv_i_imag_fun)
