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

from abc import ABCMeta
from rbnics.utils.decorators import AbstractBackend, abstractmethod

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
        
