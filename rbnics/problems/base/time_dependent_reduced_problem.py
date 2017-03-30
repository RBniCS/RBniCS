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

from math import sqrt
from rbnics.backends import assign, copy, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from rbnics.utils.decorators import Extends, override, sync_setters
from rbnics.utils.mpi import log, PROGRESS

def TimeDependentReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class TimeDependentReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        ## Default initialization of members
        @override
        @sync_setters("truth_problem", "set_time", "t")
        @sync_setters("truth_problem", "set_time_step_size", "dt")
        @sync_setters("truth_problem", "set_final_time", "T")
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
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
            self._solution_dot_cache = dict() # of Functions
            # Solution and output over time
            self._solution_over_time = list() # of Functions
            self._solution_dot_over_time = list() # of Functions
            self._solution_over_time_cache = dict() # of list of Functions
            self._solution_dot_over_time_cache = dict() # of list of Functions
            self._output_over_time = list() # of floats
            
        ## Set current time
        def set_time(self, t):
            self.t = t
            
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
            
        ## Initialize data structures required for the online phase
        def init(self, current_stage="online"):
            # Initialize first data structures related to initial conditions
            self._init_initial_condition(current_stage)
            # ... since the Parent call may be overridden to need them!
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            
        def _init_initial_condition(self, current_stage="online"):
            assert current_stage in ("online", "offline")
            n_components = len(self.components)
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
                for component in self.components:
                    try:
                        theta_ic = self.compute_theta(initial_condition_string.format(c=component))
                    except ValueError: # there were no initial condition to be imposed by lifting
                        initial_condition[component] = None
                        initial_condition_is_homogeneous[component] = True
                        Q_ic[component] = 0
                    else:
                        initial_condition_is_homogeneous[component] = False
                        Q_ic[component] = len(theta_ic)
                        if current_stage == "online":
                            initial_condition[component] = self.assemble_operator(initial_condition_string.format(c=component), "online")
                        elif current_stage == "offline":
                            initial_condition[component] = OnlineAffineExpansionStorage(Q_ic[component])
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
            n_components = len(self.components)
            if n_components > 1:
                initial_condition_string = "initial_condition_{c}"
                for component in self.components:
                    if not self.initial_condition_is_homogeneous[component]:
                        self.initial_condition[component] = self.assemble_operator(initial_condition_string.format(c=component), "offline")
            else:
                if not self.initial_condition_is_homogeneous:
                    self.initial_condition = self.assemble_operator("initial_condition", "offline")
                
        ## Assemble the reduced order affine expansion
        def assemble_operator(self, term, current_stage="online"):
            assert current_stage in ("online", "offline")
            if term.startswith("initial_condition"):
                component = term.replace("initial_condition", "").replace("_", "")
                # Compute
                if current_stage == "online": # load from file
                    initial_condition = OnlineAffineExpansionStorage(0) # it will be resized by load
                    initial_condition.load(self.folder["reduced_operators"], term)
                elif current_stage == "offline":
                # Get truth initial condition
                    if component != "":
                        truth_initial_condition = self.truth_problem.initial_condition[component]
                        truth_inner_product = self.truth_problem.inner_product[component]
                        initial_condition = self.initial_condition[component]
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
                if component != "":
                    assert component in self.components
                    self.initial_condition[component] = initial_condition
                else:
                    assert len(self.components) == 1
                    self.initial_condition = initial_condition
                # Return
                return initial_condition
            else:
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
                
        # Internal method for error computation
        @override
        def _compute_error(self, **kwargs):
            error_over_time = list()
            for (k, (truth_solution, reduced_solution)) in enumerate(zip(self.truth_problem._solution_over_time, self._solution_over_time)):
                self.set_time(k*self.dt)
                assign(self._solution, reduced_solution)
                assign(self.truth_problem._solution, truth_solution)
                error = ParametrizedReducedDifferentialProblem_DerivedClass._compute_error(self, **kwargs)
                error_over_time.append(error)
            return error_over_time
            
        # Internal method for relative error computation
        @override
        def _compute_relative_error(self, absolute_error_over_time, **kwargs):
            relative_error_over_time = list()
            for (k, (truth_solution, absolute_error)) in enumerate(zip(self.truth_problem._solution_over_time, absolute_error_over_time)):
                self.set_time(k*self.dt)
                assign(self.truth_problem._solution, truth_solution)
                if absolute_error != 0.0:
                    relative_error = ParametrizedReducedDifferentialProblem_DerivedClass._compute_relative_error(self, absolute_error, **kwargs)
                    relative_error_over_time.append(relative_error)
                else:
                    relative_error_over_time.append(0.0)
            return relative_error_over_time
            
        # Internal method for output error computation
        @override
        def _compute_error_output(self, **kwargs):
            error_output_over_time = list()
            for (k, (truth_output, reduced_output)) in enumerate(zip(self.truth_problem._output_over_time, self._output_over_time)):
                self.set_time(k*self.dt)
                self._output = reduced_output
                self.truth_problem._output = truth_output
                error_output = ParametrizedReducedDifferentialProblem_DerivedClass._compute_error_output(self, **kwargs)
                error_output_over_time.append(error_output)
            return error_output_over_time
            
        # Internal method for output relative error computation
        @override
        def _compute_relative_error_output(self, absolute_error_output_over_time, **kwargs):
            relative_error_output_over_time = list()
            for (k, (truth_output, absolute_error_output)) in enumerate(zip(self.truth_problem._output_over_time, absolute_error_output_over_time)):
                self.set_time(k*self.dt)
                self.truth_problem._output = truth_output
                if absolute_error_output != 0.0:
                    relative_error_output = ParametrizedReducedDifferentialProblem_DerivedClass._compute_relative_error_output(self, absolute_error_output, **kwargs)
                    relative_error_output_over_time.append(relative_error_output)
                else:
                    relative_error_output_over_time.append(0.0)
            return relative_error_output_over_time
        
        ## Export solution to file
        @override
        def export_solution(self, folder, filename, solution_over_time=None, solution_dot_over_time=None, component=None, suffix=None):
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            if solution_dot_over_time is None:
                solution_dot_over_time = self._solution_dot_over_time
            solution_over_time_as_truth_function = list()
            solution_dot_over_time_as_truth_function = list()
            for (k, (solution, solution_dot)) in enumerate(zip(solution_over_time, solution_dot_over_time)):
                N = solution.N
                solution_over_time_as_truth_function.append(self.Z[:N]*solution)
                solution_dot_over_time_as_truth_function.append(self.Z[:N]*solution_dot)
            self.truth_problem.export_solution(folder, filename, solution_over_time_as_truth_function, solution_dot_over_time_as_truth_function, component, suffix)
            
        @override
        def solve(self, N=None, **kwargs):
            N, kwargs = self._online_size_from_kwargs(N, **kwargs)
            N += self.N_bc
            cache_key = self._cache_key_from_N_and_kwargs(N, **kwargs)
            self._solution = OnlineFunction(N)
            self._solution_dot = OnlineFunction(N)
            if cache_key in self._solution_cache:
                log(PROGRESS, "Loading reduced solution from cache")
                assert cache_key in self._solution_dot_cache
                assert cache_key in self._solution_over_time_cache
                assert cache_key in self._solution_dot_over_time_cache
                assign(self._solution, self._solution_cache[cache_key])
                assign(self._solution_dot, self._solution_dot_cache[cache_key])
                assign(self._solution_over_time, self._solution_over_time_cache[cache_key])
                assign(self._solution_dot_over_time, self._solution_dot_over_time_cache[cache_key])
            else:
                log(PROGRESS, "Solving reduced problem")
                assert not hasattr(self, "_is_solving")
                self._is_solving = True
                self._solve(N, **kwargs)
                delattr(self, "_is_solving")
                self._solution_cache[cache_key] = copy(self._solution)
                self._solution_dot_cache[cache_key] = copy(self._solution_dot)
                self._solution_over_time_cache[cache_key] = copy(self._solution_over_time)
                self._solution_dot_over_time_cache[cache_key] = copy(self._solution_dot_over_time)
            return self._solution_over_time
                
        # Perform an online evaluation of the output
        @override
        def compute_output(self):
            N = self._solution.N
            self._compute_output(N)
            return self._output_over_time
            
        # Perform an online evaluation of the output. Internal method
        @override
        def _compute_output(self, N):
            self._output_over_time = [NotImplemented]*len(self._solution_over_time)
            self._output = NotImplemented
        
    # return value (a class) for the decorator
    return TimeDependentReducedProblem_Class
    
