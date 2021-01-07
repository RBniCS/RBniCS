# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class NonlinearSolver(object, metaclass=ABCMeta):
    def __init__(self, problem_wrapper, solution):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass

    @abstractmethod
    def solve(self):
        pass


class NonlinearProblemWrapper(object, metaclass=ABCMeta):
    @abstractmethod
    def jacobian_eval(self, solution):
        pass

    @abstractmethod
    def residual_eval(self, solution):
        pass

    @abstractmethod
    def bc_eval(self):
        pass

    @abstractmethod
    def monitor(self, solution):
        pass
