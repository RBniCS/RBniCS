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
class TimeStepping(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, problem_wrapper, solution, solution_dot, solution_dot_dot=None):
        pass
        
    @abstractmethod
    def set_parameters(self, parameters):
        pass
        
    @abstractmethod
    def solve(self):
        pass

class TimeDependentProblemWrapper(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def time_order(self):
        pass
    
    @abstractmethod
    def bc_eval(self, t):
        pass
        
    @abstractmethod
    def ic_eval(self):
        pass

class TimeDependentProblem1Wrapper(TimeDependentProblemWrapper):
    def time_order(self):
        return 1
        
    @abstractmethod
    def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
        pass
        
    @abstractmethod
    def residual_eval(self, t, solution, solution_dot):
        pass

class TimeDependentProblem2Wrapper(TimeDependentProblemWrapper):
    def time_order(self):
        return 2
        
    @abstractmethod
    def jacobian_eval(self, t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient):
        pass
        
    @abstractmethod
    def residual_eval(self, t, solution, solution_dot, solution_dot_dot):
        pass
        
