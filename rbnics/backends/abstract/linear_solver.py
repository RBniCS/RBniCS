# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class LinearSolver(object, metaclass=ABCMeta):
    # will use @overload in derived classes
    def __init__(self, lhs, solution, rhs, bcs=None):
        pass

    # will use @overload in derived classes
    def __init__(self, problem_wrapper, solution):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass

    @abstractmethod
    def solve(self):
        pass


class LinearProblemWrapper(object, metaclass=ABCMeta):
    @abstractmethod
    def matrix_eval(self):
        pass

    @abstractmethod
    def vector_eval(self):
        pass

    @abstractmethod
    def bc_eval(self):
        pass

    @abstractmethod
    def monitor(self, solution):
        pass
