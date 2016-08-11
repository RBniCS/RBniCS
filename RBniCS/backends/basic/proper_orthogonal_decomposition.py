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
## @file proper_orthogonal_decomposition.py
#  @brief Implementation of the POD
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from math import sqrt
from numpy import isclose, zeros, sum as total_energy, cumsum as retained_energy
from RBniCS.backends.abstract import ProperOrthogonalDecomposition as AbstractProperOrthogonalDecomposition
from RBniCS.backends.online import OnlineEigenSolver
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.mpi.mpi import mpi_comm
from RBniCS.utils.mpi.print import print

# Class containing the implementation of the POD
@Extends(AbstractProperOrthogonalDecomposition)
class ProperOrthogonalDecomposition(AbstractProperOrthogonalDecomposition):

    @override
    def __init__(self, X, V_or_Z, backend):
        self.X = X
        self.backend = backend
        self.V_or_Z = V_or_Z
        
        # Declare a matrix to store the snapshots
        self.snapshots_matrix = backend.SnapshotsMatrix(self.V_or_Z)
        # Declare the eigen solver to compute the POD
        self.eigensolver = OnlineEigenSolver()
        # Store inner product
        self.X = X
        
    @override
    def clear(self):
        self.snapshots_matrix.clear()
        self.eigensolver = OnlineEigenSolver()
        
    @override
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
            
    @override
    def apply(self, Nmax):
        assert len(self.X) == 1 # note that we cannot move this assert in __init__ because
                                # self.X has not been assembled yet there
        X = self.X[0]
        snapshots_matrix = self.snapshots_matrix
        transpose = self.backend.transpose
        
        correlation = transpose(snapshots_matrix)*X*snapshots_matrix
        
        eigensolver = OnlineEigenSolver(correlation)
        parameters["problem_type"] = "hermitian"
        parameters["spectrum"] = "largest real"
        eigensolver.set_parameters(parameters)
        eigensolver.solve()
        
        Z = backend.BasisFunctionsMatrix(self.V_or_Z)
        for i in range(Nmax):
            (eigvector, _) = eigensolver.get_eigenvector(Nmax)
            b = self.snapshots_matrix*eigvector
            b /= sqrt(transpose(b)*X*b)
            Z.enrich(b)
            
        self.eigensolver = eigensolver
        return (Z, Nmax)

    @override
    def print_eigenvalues(self, N=None):
        if N is None:
            N = len(self.snapshots_matrix)
        for i in range(N):
            (eig_i_real, eig_i_complex) = self.eigensolver.get_eigenvalue[i]
            assert isclose(eig_i_complex, 0)
            print("lambda_" + str(i) + " = " + str(eig_i_real))
        
    @override
    def save_eigenvalues_file(self, output_directory, eigenvalues_file):
        if mpi_comm.rank == 0:
            with open(str(directory) + "/" + filename, "w") as outfile:
                N = len(self.snapshots_matrix)
                for i in range(N):
                    (eig_i_real, eig_i_complex) = self.eigensolver.get_eigenvalue[i]
                    assert isclose(eig_i_complex, 0)
                    outfile.write(str(i) + " " + str(eig_i_real) + "\n")
        mpi_comm.barrier()
        
    @override
    def save_retained_energy_file(self, output_directory, retained_energy_file):
        if mpi_comm.rank == 0:
            N = len(self.snapshots_matrix)
            eigs = zeros(N)
            for i in range(N):
                (eigs[i], _) = self.eigensolver.get_eigenvalue[i]
            energy = total_energy(eigs)
            retained_energy = retained_energy(eigs)
            retained_energy /= energy
            with open(str(directory) + "/" + filename, "w") as outfile:
                for i in range(N):
                    outfile.write(str(i) + " " + str(retained_energy[i]) + "\n") 
        mpi_comm.barrier()
    
