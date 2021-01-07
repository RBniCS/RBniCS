# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class TimeStepping(object, metaclass=ABCMeta):
    def __init__(self, problem_wrapper, solution, solution_dot, solution_dot_dot=None):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass

    @abstractmethod
    def solve(self):
        pass


class TimeDependentProblemWrapper(object, metaclass=ABCMeta):
    def set_time(self, t):
        pass

    @abstractmethod
    def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
        pass

    @abstractmethod
    def residual_eval(self, t, solution, solution_dot):
        pass

    @abstractmethod
    def bc_eval(self, t):
        pass

    @abstractmethod
    def ic_eval(self):
        pass

    @abstractmethod
    def monitor(self, t, solution, solution_dot):
        pass
