# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from numpy import abs, cumsum as compute_retained_energy, isclose, sum as compute_total_energy
from rbnics.utils.io import ExportableList


# Class containing the implementation of the POD
def ProperOrthogonalDecompositionBase(backend, wrapping, online_backend, online_wrapping,
                                      ParentProperOrthogonalDecomposition, SnapshotsContainerType,
                                      BasisContainerType):

    class _ProperOrthogonalDecompositionBase(ParentProperOrthogonalDecomposition):

        def __init__(self, space, inner_product, *args):
            self.inner_product = inner_product
            self.space = space
            self.args = args

            # Declare a matrix to store the snapshots
            self.snapshots_matrix = SnapshotsContainerType(self.space, *args)
            # Declare a list to store eigenvalues
            self.eigenvalues = ExportableList("text")
            self.retained_energy = ExportableList("text")

        def clear(self):
            self.snapshots_matrix.clear()
            self.eigenvalues = ExportableList("text")
            self.retained_energy = ExportableList("text")

        # No implementation is provided for store_snapshot, because
        # it has different interface for the standard POD and
        # the tensor one.

        def apply(self, Nmax, tol):
            inner_product = self.inner_product
            snapshots_matrix = self.snapshots_matrix
            transpose = backend.transpose

            if inner_product is not None:
                correlation = transpose(snapshots_matrix) * inner_product * snapshots_matrix
            else:
                correlation = transpose(snapshots_matrix) * snapshots_matrix

            basis_functions = BasisContainerType(self.space, *self.args)

            eigensolver = online_backend.OnlineEigenSolver(basis_functions, correlation)
            parameters = {
                "problem_type": "hermitian",
                "spectrum": "largest real"
            }
            eigensolver.set_parameters(parameters)
            eigensolver.solve()

            Neigs = len(self.snapshots_matrix)
            Nmax = min(Nmax, Neigs)
            assert len(self.eigenvalues) == 0
            for i in range(Neigs):
                (eig_i_real, eig_i_complex) = eigensolver.get_eigenvalue(i)
                assert isclose(eig_i_complex, 0.)
                self.eigenvalues.append(eig_i_real)

            total_energy = compute_total_energy([abs(e) for e in self.eigenvalues])
            retained_energy = compute_retained_energy([abs(e) for e in self.eigenvalues])
            assert len(self.retained_energy) == 0
            if total_energy > 0.:
                self.retained_energy.extend([retained_energy_i / total_energy
                                             for retained_energy_i in retained_energy])
            else:
                self.retained_energy.extend([1. for _ in range(Neigs)])  # trivial case, all snapshots are zero

            eigenvectors = list()
            for N in range(Nmax):
                (eigvector, _) = eigensolver.get_eigenvector(N)
                eigenvectors.append(eigvector)
                b = self.snapshots_matrix * eigvector
                if inner_product is not None:
                    norm_b = sqrt(transpose(b) * inner_product * b)
                else:
                    norm_b = sqrt(transpose(b) * b)
                if norm_b != 0.:
                    b /= norm_b
                basis_functions.enrich(b)
                if tol > 0. and self.retained_energy[N] > 1. - tol:
                    break
            N += 1

            return (self.eigenvalues[:N], eigenvectors, basis_functions, N)

        def print_eigenvalues(self, N=None):
            if N is None:
                N = len(self.snapshots_matrix)
            for i in range(N):
                print("lambda_" + str(i) + " = " + str(self.eigenvalues[i]))

        def save_eigenvalues_file(self, output_directory, eigenvalues_file):
            self.eigenvalues.save(output_directory, eigenvalues_file)

        def save_retained_energy_file(self, output_directory, retained_energy_file):
            self.retained_energy.save(output_directory, retained_energy_file)

    return _ProperOrthogonalDecompositionBase
