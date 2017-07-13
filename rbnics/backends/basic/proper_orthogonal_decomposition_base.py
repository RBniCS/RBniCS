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

from __future__ import print_function
from math import sqrt
from numpy import abs, isclose, zeros, sum as compute_total_energy, cumsum as compute_retained_energy
from rbnics.backends.abstract import ProperOrthogonalDecomposition as AbstractProperOrthogonalDecomposition
import rbnics.backends.online
from rbnics.utils.decorators import Extends, override
from rbnics.utils.mpi import is_io_process, print

# Class containing the implementation of the POD
def ProperOrthogonalDecompositionBase(ParentProperOrthogonalDecomposition):
    @Extends(ParentProperOrthogonalDecomposition)
    class ProperOrthogonalDecompositionBase(ParentProperOrthogonalDecomposition):

        @override
        def __init__(self, V_or_Z, X, container_type_second_argument, backend, wrapping, SnapshotsContainerType, BasisContainerType):
            self.X = X
            self.backend = backend
            self.BasisContainerType = BasisContainerType
            self.V_or_Z = V_or_Z
            self.container_type_second_argument = container_type_second_argument
            self.mpi_comm = wrapping.get_mpi_comm(V_or_Z)
            
            # Declare a matrix to store the snapshots
            if self.container_type_second_argument is None:
                self.snapshots_matrix = SnapshotsContainerType(self.V_or_Z)
            else:
                self.snapshots_matrix = SnapshotsContainerType(self.V_or_Z, self.container_type_second_argument)
            # Declare a list to store eigenvalues
            self.eigenvalues = zeros(0) # correct size will be assigned later
            self.retained_energy = zeros(0) # correct size will be assigned later
            # Store inner product
            self.X = X
            
        @override
        def clear(self):
            self.snapshots_matrix.clear()
            self.eigenvalues = zeros(0)
            
        # No implementation is provided for store_snapshot, because
        # it has different interface for the standard POD and
        # the tensor one.
                
        @override
        def apply(self, Nmax, tol):
            X = self.X
            snapshots_matrix = self.snapshots_matrix
            transpose = self.backend.transpose
            
            if X is not None:
                correlation = transpose(snapshots_matrix)*X*snapshots_matrix
            else:
                correlation = transpose(snapshots_matrix)*snapshots_matrix
            
            if self.container_type_second_argument is None:
                Z = self.BasisContainerType(self.V_or_Z)
            else:
                Z = self.BasisContainerType(self.V_or_Z, self.container_type_second_argument)
                            
            eigensolver = rbnics.backends.online.OnlineEigenSolver(Z, correlation) 
            parameters = {
                "problem_type": "hermitian",
                "spectrum": "largest real"
            }
            eigensolver.set_parameters(parameters)
            eigensolver.solve()
            
            Neigs = len(self.snapshots_matrix)
            Nmax = min(Nmax, Neigs)
            self.eigenvalues = zeros(Neigs)
            for i in range(Neigs):
                (eig_i_real, eig_i_complex) = eigensolver.get_eigenvalue(i)
                assert isclose(eig_i_complex, 0.)
                self.eigenvalues[i] = eig_i_real
            
            total_energy = compute_total_energy(abs(self.eigenvalues))
            self.retained_energy = compute_retained_energy(abs(self.eigenvalues))
            if total_energy > 0.:
                self.retained_energy /= total_energy
            else:
                self.retained_energy += 1. # trivial case, all snapshots are zero
            
            for N in range(Nmax):
                (eigvector, _) = eigensolver.get_eigenvector(N)
                b = self.snapshots_matrix*eigvector
                if X is not None:
                    norm_b = sqrt(transpose(b)*X*b)
                else:
                    norm_b = sqrt(transpose(b)*b)
                if norm_b != 0.:
                    b /= norm_b
                Z.enrich(b)
                if self.retained_energy[N] > 1. - tol:
                    break
            N += 1
            
            return (self.eigenvalues[:N], Z, N)
                
        @override
        def print_eigenvalues(self, N=None):
            if N is None:
                N = len(self.snapshots_matrix)
            for i in range(N):
                print("lambda_" + str(i) + " = " + str(self.eigenvalues[i]))
            
        @override
        def save_eigenvalues_file(self, output_directory, eigenvalues_file):
            if is_io_process(self.mpi_comm):
                N = len(self.snapshots_matrix)
                with open(str(output_directory) + "/" + eigenvalues_file, "w") as outfile:
                    for i in range(N):
                        outfile.write(str(i) + " " + str(self.eigenvalues[i]) + "\n")
            self.mpi_comm.barrier()
            
        @override
        def save_retained_energy_file(self, output_directory, retained_energy_file):
            if is_io_process(self.mpi_comm):
                N = len(self.snapshots_matrix)
                with open(str(output_directory) + "/" + retained_energy_file, "w") as outfile:
                    for i in range(N):
                        outfile.write(str(i) + " " + str(self.retained_energy[i]) + "\n") 
            self.mpi_comm.barrier()
    
    return ProperOrthogonalDecompositionBase
    
