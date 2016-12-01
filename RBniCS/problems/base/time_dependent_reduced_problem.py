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

from math import sqrt
from RBniCS.backends import assign, TimeQuadrature, transpose
from RBniCS.backends.online import OnlineAffineExpansionStorage, OnlineFunction
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
            # Online reduced space dimension
            self.initial_condition = None # bool (for problems with one component) or dict of bools (for problem with several components)
            self.initial_condition_is_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
            # Number of terms in the affine expansion
            self.Q_ic = None # integer (for problems with one component) or dict of integers (for problem with several components)
            # Time derivative of the solution, at the current time
            self._solution_dot = OnlineFunction()
            # Solution and output over time
            self._solution_over_time = list() # of Functions
            self._solution_dot_over_time = list() # of Functions
            self._output_over_time = list() # of floats
            
        ## Initialize data structures required for the online phase
        def init(self, current_stage="online"):
            # Initialize first data structures related to initial conditions
            self._init_initial_condition(current_stage)
            # ... since the Parent call may be overridden to need them!
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            
        def _init_initial_condition(self, current_stage="online"):
            assert current_stage in ("online", "offline")
            n_components = len(self.truth_problem.components_name)
            # Get helper strings depending on the number of components
            if n_components > 1:
                initial_condition_string = "initial_condition_{c}"
            else:
                initial_condition_string = "initial_condition"
            # Detect how many theta terms are related to boundary conditions
            # we do not assert for
            # (self.initial_condition is None) == (self.initial_condition_is_homogeneous is None)
            # because self.initial_condition may still be None after initialization, if there
            # were no initial condition at all and the problem had only one component
            if self.initial_condition_is_homogeneous is None: # init was not called already
                initial_condition = dict()
                initial_condition_is_homogeneous = dict()
                Q_ic = dict()
                for (component_index, component_name) in enumerate(self.truth_problem.components_name):
                    try:
                        theta_ic = self.compute_theta(initial_condition_string.format(c=component_name))
                    except ValueError: # there were no initial condition to be imposed by lifting
                        initial_condition[component_name] = None
                        initial_condition_is_homogeneous[component_name] = True
                        Q_ic[component_name] = 0
                    else:
                        initial_condition_is_homogeneous[component_name] = False
                        Q_ic[component_name] = len(theta_ic)
                        if current_stage == "online":
                            initial_condition[component_name] = self.assemble_operator(initial_condition_string.format(c=component_name), "online")
                        elif current_stage == "offline":
                            initial_condition[component_name] = OnlineAffineExpansionStorage(Q_ic[component_name])
                        else:
                            raise AssertionError("Invalid stage in _init_initial_condition().")
                if n_components == 1:
                    self.initial_condition = initial_condition.values()[0]
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous.values()[0]
                    self.Q_ic = Q_ic.values()[0]
                else:
                    self.initial_condition = initial_condition
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous
                    self.Q_ic = Q_ic
                assert self.initial_condition_is_homogeneous == self.truth_problem.initial_condition_is_homogeneous
                
        ## Assemble the reduced order affine expansion.
        def build_reduced_operators(self):
            ParametrizedReducedDifferentialProblem_DerivedClass.build_reduced_operators(self)
            # Initial condition
            n_components = len(self.truth_problem.components_name)
            if n_components > 1:
                initial_condition_string = "initial_condition_{c}"
                for (component_index, component_name) in enumerate(self.truth_problem.components_name):
                    if not self.initial_condition_is_homogeneous[component_name]:
                        self.initial_condition[component_name] = self.assemble_operator(initial_condition_string.format(c=component_name), "offline")
            else:
                if not self.initial_condition_is_homogeneous:
                    self.initial_condition = self.assemble_operator("initial_condition", "offline")
                
        ## Assemble the reduced order affine expansion
        def assemble_operator(self, term, current_stage="online"):
            assert current_stage in ("online", "offline")
            if term.startswith("initial_condition"):
                component_name = term.replace("initial_condition", "").replace("_", "")
                # Compute
                if current_stage == "online": # load from file
                    initial_condition = OnlineAffineExpansionStorage(0) # it will be resized by load
                    initial_condition.load(self.folder["reduced_operators"], term)
                elif current_stage == "offline":
                # Get truth initial condition
                    if component_name != "":
                        truth_initial_condition = self.truth_problem.initial_condition[component_name]
                        truth_inner_product = self.truth_problem.inner_product[component_name]
                        initial_condition = self.initial_condition[component_name]
                    else:
                        truth_initial_condition = self.truth_problem.initial_condition
                        truth_inner_product = self.truth_problem.inner_product
                        initial_condition = self.initial_condition
                    assert len(truth_inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                    for (q, truth_initial_condition_q) in enumerate(truth_initial_condition):
                        initial_condition[q] = transpose(self.Z)*truth_inner_product[0]*truth_initial_condition_q
                    initial_condition.save(self.folder["reduced_operators"], term)
                else:
                    raise AssertionError("Invalid stage in assemble_operator().")
                # Assign
                if component_name != "":
                    assert component_name in self.truth_problem.components_name
                    self.initial_condition[component_name] = initial_condition
                else:
                    assert len(self.truth_problem.components_name) == 1
                    self.initial_condition = initial_condition
                # Return
                return initial_condition
            else:
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
                
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
                    errors_over_time[tuple_index].append(error**2)
            time_quadrature = TimeQuadrature((0., self.T), self.dt)
            integrated_errors_over_time = list() # of real numbers
            for (tuple_index, error_over_time) in enumerate(errors_over_time):
                integrated_errors_over_time.append( sqrt(time_quadrature.integrate(error_over_time)) )
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
    
