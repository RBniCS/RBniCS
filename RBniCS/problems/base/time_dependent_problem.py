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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends import AffineExpansionStorage, export, Function
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
            self._time_stepping_parameters["initial_time"] = 0.
            # Matrices/vectors resulting from the truth discretization
            self.initial_condition = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components)
            self.initial_condition_is_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
            # Time derivative of the solution, at the current time
            self._solution_dot = Function(self.V)
            # Solution and output over time
            self._solution_over_time = list() # of Functions
            self._solution_dot_over_time = list() # of Functions
            self._output_over_time = list() # of floats
            
        ## Set time step size
        def set_time_step_size(self, dt):
            assert isinstance(dt, (float, int))
            dt = float(dt)
            self.dt = dt
            self._time_stepping_parameters["time_step_size"] = dt
            
        ## Set final time
        def set_final_time(self, T):
            assert isinstance(T, (float, int))
            T = float(T)
            self.T = T
            self._time_stepping_parameters["final_time"] = T
            
        ## Export solution to file
        @override
        def export_solution(self, folder, filename, solution_over_time=None, component=None):
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            for (k, solution) in enumerate(solution_over_time):
                ParametrizedDifferentialProblem_DerivedClass.export_solution(self, folder, filename, solution, component)
                
        ## Initialize data structures required for the offline phase
        @override
        def init(self):
            ParametrizedDifferentialProblem_DerivedClass.init(self)
            self._init_initial_condition()
            
        def _init_initial_condition(self):
            # Get helper strings depending on the number of basis components
            n_components = len(self.components)
            assert n_components > 0
            if n_components > 1:
                initial_condition_string = "initial_condition_{c}"
            else:
                initial_condition_string = "initial_condition"
            # Assemble initial condition
            # we do not assert for
            # (self.initial_condition is None) == (self.initial_condition_is_homogeneous is None)
            # because self.initial_condition may still be None after initialization, if there
            # were no initial condition at all and the problem had only one component
            if self.initial_condition_is_homogeneous is None: # init was not called already
                initial_condition = dict()
                initial_condition_is_homogeneous = dict()
                for component in self.components:
                    try:
                        operator_ic = AffineExpansionStorage(self.assemble_operator(initial_condition_string.format(c=component)))
                    except ValueError: # there were no initial condition: assume homogeneous one
                        initial_condition[component] = None
                        initial_condition_is_homogeneous[component] = True
                    else:
                        initial_condition[component] = operator_ic
                        initial_condition_is_homogeneous[component] = False
                if n_components == 1:
                    self.initial_condition = initial_condition.values()[0]
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous.values()[0]
                else:
                    self.initial_condition = initial_condition
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous
                
    # return value (a class) for the decorator
    return TimeDependentProblem_Class
    
