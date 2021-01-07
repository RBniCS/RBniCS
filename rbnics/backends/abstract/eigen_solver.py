# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class EigenSolver(object, metaclass=ABCMeta):
    def __init__(self, space, A, B=None, bcs=None):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass

    @abstractmethod
    def solve(self, n_eigs=None):
        pass

    @abstractmethod
    def get_eigenvalue(self, i):
        pass

    @abstractmethod
    def get_eigenvector(self, i):
        pass
