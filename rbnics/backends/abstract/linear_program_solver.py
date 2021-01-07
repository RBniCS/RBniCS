# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class LinearProgramSolver(object, metaclass=ABCMeta):
    def __init__(self, cost, inequality_constraints_matrix, inequality_constraints_vector, bounds):
        """
        Solve the linear program
            min     c^T x
            s.t.    A x >= b
                    x_{min} <= x <= x_{max}
        where
            c                   is the first input parameter
            A                   is the second input parameter
            b                   is the third input parameter
           (x_{min}, x_{max})   are given as a list of (min, max) tuples in the fourth input parameter
        """
        pass

    @abstractmethod
    def solve(self):
        pass
