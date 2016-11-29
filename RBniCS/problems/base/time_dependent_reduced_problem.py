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

from RBniCS.backends import assign, TimeQuadrature
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override

def TimeDependentReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class TimeDependentReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        ## Default initialization of members
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem)
            # Store quantities related to the time discretization
            assert truth_problem.t == 0.
            self.t = 0.
            assert truth_problem.dt is not None
            self.dt = truth_problem.dt
            assert truth_problem.T is not None
            self.T  = truth_problem.T
            # Additional options for time stepping may be stored in the following dict
            self._time_stepping_parameters = dict()
            self._time_stepping_parameters["time_step_size"] = self.dt
            self._time_stepping_parameters["final_time"] = self.T
            # Time derivative of the solution, at the current time
            self._solution_dot = OnlineFunction()
            # Solution and output over time
            self._solution_over_time = list() # of Functions
            self._solution_dot_over_time = list() # of Functions
            self._output_over_time = list() # of floats
            
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
            
        # Internal method for error computation
        def _compute_error(self):
            errors_over_time = list() # (over compute_error tuple output index) of list (over time) of real numbers
            for (k, (truth_solution, reduced_solution)) in enumerate(zip(self.truth_problem._solution_over_time, self._solution_over_time)):
                self.t = k*self.dt
                assign(self._solution, reduced_solution)
                self.truth_problem.t = k*self.dt
                assign(self.truth_problem._solution, truth_solution)
                errors = ParametrizedReducedDifferentialProblem_DerivedClass._compute_error(self)
                if len(errors_over_time) == 0:
                    errors_over_time = [list() for _ in range(len(errors))]
                for (tuple_index, error) in enumerate(errors):
                    errors_over_time[tuple_index].append(error)
            time_quadrature = TimeQuadrature((0., self.T), self.dt)
            integrated_errors_over_time = list() # of real numbers
            for (tuple_index, error_over_time) in enumerate(errors_over_time):
                integrated_errors_over_time.append( time_quadrature.integrate(error_over_time) )
            return integrated_errors_over_time
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ###########################
        
        ## Export solution to file
        @override
        def export_solution(self, folder, filename, solution_over_time=None, component=None):
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            solution_over_time_as_truth_function = list()
            for (k, solution) in enumerate(solution_over_time):
                N = solution.N
                solution_over_time_as_truth_function.append(self.Z[:N]*solution)
            self.truth_problem.export_solution(folder, filename, solution_over_time_as_truth_function, component)
        
    # return value (a class) for the decorator
    return TimeDependentReducedProblem_Class
    
