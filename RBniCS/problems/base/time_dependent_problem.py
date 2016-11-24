# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends import Function
from RBniCS.utils.decorators import Extends, override

def TimeDependentProblem(ParametrizedDifferentialProblem_DerivedClass):
    
    @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class TimeDependentProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        
        ## Default initialization of members
        @override
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
            # Store quantities related to the time discretization
            self.t = 0.
            self.dt = None
            self.T  = None
            # Additional options for time stepping may be stored in the following dict
            self._time_stepping_parameters = dict()
            # Time derivative of the solution, at the current time
            self._solution_dot = Function(self.V)
            # Solution and output over time
            self._solution_over_time = list() # of Functions
            self._output_over_time = list() # of floats
            
        ## Set time step size
        def set_time_step_size(self, dt):
            self.dt = dt
            self._time_stepping_parameters["time_step_size"] = dt
            
        ## Set final time
        def set_final_time(self, T):
            self.T = T
            self._time_stepping_parameters["final_time"] = T
            
        ## Export solution to file
        def export_solution(self, folder, filename, solution_over_time=None, component=None):
            if solution is None:
                solution_over_time = self._solution_over_time
            for (k, solution) in enumerate(solution_over_time):
                export(solution, folder, filename, component, suffix=k)
                
    # return value (a class) for the decorator
    return TimeDependentProblem_Class
    
