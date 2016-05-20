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

from __future__ import print_function

###########################     ONLINE STAGE     ########################### 
## @defgroup OnlineStage Methods related to the online stage
#  @{

# Declare reduced eigen solver type
from numpy import real
from numpy.linalg import eig as OnlineEigenSolver_Impl

class OnlineEigenSolver(object):
    def __init__(self, A = None, B = None):
        if A is not None:
            assert A.shape[0] == A.shape[1]
        if B is not None:
            assert B.shape[0] == B.shape[1]
            assert A.shape[0] == B.shape[0]
        
        self.A = A
        self.B = B
        self.parameters = {}
        
    def solve(self):
        assert self.parameters["problem_type"] == "hermitian" # only one implemented so far
        assert self.parameters["spectrum"] == "largest real" # only one implemented so far
        assert self.A is not None
        assert self.B is None # generalized version not implemented so far
        
        eigs, eigv = OnlineEigenSolver_Impl(self.A)
        
        idx = eigs.argsort()
        idx = idx[::-1]
        eigs = eigs[idx]
        eigv = eigv[:, idx]
        
        # Remove (negigible) complex parts
        self.eigs = real(eigs)
        self.eigv = real(eigv) # is a matrix because A and B were matrices
    
    def get_eigenvalue(self, i):
        return self.eigs[i]
        
    def get_eigenvalues(self, i_max):
        return self.eigs[:i_max]
        
    def get_eigenvector(self, i):
        return self.eigv[:, i]
        
    def get_eigenvectors(self, i_max):
        return self.eigv[:, :i_max]
    
    def print_eigenvalues(self):
        for i in range(len(self.eigs)):
            print("lambda_" + str(i) + " = " + str(self.eigs[i]))
    
    def save_eigenvalues_file(self, directory, filename):
        with open(directory + "/" + filename, "a") as outfile:
            for i in range(len(self.eigs)):
                outfile.write(str(i) + " " + str(self.eigs[i]) + "\n")
    
    def save_retained_energy_file(self, directory, filename):
        from numpy import sum as np_sum
        from numpy import cumsum as np_cumsum
        energy = np_sum(self.eigs)
        eigs_cumsum = np_cumsum(self.eigs)
        eigs_cumsum /= energy
        with open(directory + "/" + filename, "a") as outfile:
            for i in range(len(eigs_cumsum)):
                file.write(str(i) + " " + str(eigs_cumsum[i]) + "\n") 
    
#  @}
########################### end - ONLINE STAGE - end ########################### 

