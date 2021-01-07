# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


# Class containing the implementation of the POD
@AbstractBackend
class HighOrderProperOrthogonalDecomposition(object, metaclass=ABCMeta):
    def __init__(self, space):
        pass

    # Clean up
    @abstractmethod
    def clear(self):
        pass

    # Store a snapshot in the snapshot matrix
    @abstractmethod
    def store_snapshot(self, snapshot):
        pass

    # Perform POD on the snapshots previously computed, and store the first
    # POD modes in the basis functions matrix.
    # Input arguments are: Nmax, tol
    # Output arguments are: POD eigenvalues, POD modes, number of POD modes
    @abstractmethod
    def apply(self, Nmax, tol):
        pass

    @abstractmethod
    def print_eigenvalues(self, N=None):
        pass

    @abstractmethod
    def save_eigenvalues_file(self, output_directory, eigenvalues_file):
        pass

    @abstractmethod
    def save_retained_energy_file(self, output_directory, retained_energy_file):
        pass
