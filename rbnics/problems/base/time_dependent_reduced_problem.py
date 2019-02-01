# Copyright (C) 2015-2019 by the RBniCS authors
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

from numbers import Number
from numpy import isclose
from rbnics.backends import assign, copy, product, sum, TimeDependentProblemWrapper, TimeSeries, TimeQuadrature, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction, OnlineLinearSolver, OnlineTimeStepping
from rbnics.utils.cache import Cache, TimeSeriesCache
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators, sync_setters

@RequiredBaseDecorators(None)
def TimeDependentReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class TimeDependentReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        # Default initialization of members
        @sync_setters("truth_problem", "set_time", "t")
        @sync_setters("truth_problem", "set_initial_time", "t0")
        @sync_setters("truth_problem", "set_time_step_size", "dt")
        @sync_setters("truth_problem", "set_final_time", "T")
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Store quantities related to the time discretization
            assert truth_problem.t == 0.
            self.t = 0.
            self.t0 = truth_problem.t0
            assert truth_problem.dt is not None
            self.dt = truth_problem.dt
            assert truth_problem.T is not None
            self.T = truth_problem.T
            # Additional options for time stepping may be stored in the following dict
            self._time_stepping_parameters = dict()
            self._time_stepping_parameters["initial_time"] = self.t0
            self._time_stepping_parameters["time_step_size"] = self.dt
            self._time_stepping_parameters["final_time"] = self.T
            if "monitor" in truth_problem._time_stepping_parameters:
                self._time_stepping_parameters["monitor"] = truth_problem._time_stepping_parameters["monitor"]
            # Online reduced space dimension
            self.initial_condition = None # bool (for problems with one component) or dict of bools (for problem with several components)
            self.initial_condition_is_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
            # Number of terms in the affine expansion
            self.Q_ic = None # integer (for problems with one component) or dict of integers (for problem with several components)
            # Time derivative of the solution, at the current time
            self._solution_dot = None # OnlineFunction
            # Solution and output over time
            self._solution_over_time = None # TimeSeries of Functions
            self._solution_dot_over_time = None # TimeSeries of Functions
            self._output_over_time = None # TimeSeries of numbers
            # I/O
            def _solution_cache_key_generator(*args, **kwargs):
                assert len(args) == 2
                assert args[0] == self.mu
                return self._cache_key_from_N_and_kwargs(args[1], **kwargs)
            self._solution_over_time_cache = TimeSeriesCache(
                "reduced problems",
                key_generator=_solution_cache_key_generator
            )
            self._solution_dot_over_time_cache = TimeSeriesCache(
                "reduced problems",
                key_generator=_solution_cache_key_generator
            )
            del self._solution_cache
            def _output_cache_key_generator(*args, **kwargs):
                assert len(args) == 2
                assert args[0] == self.mu
                return self._cache_key_from_N_and_kwargs(args[1], **kwargs)
            self._output_over_time_cache = Cache(
                "reduced problems",
                key_generator=_output_cache_key_generator
            )
            del self._output_cache
            
        # Set current time
        def set_time(self, t):
            self.t = t
            
        # Set initial time
        def set_initial_time(self, t0):
            assert isinstance(t0, Number)
            self.t0 = t0
            self._time_stepping_parameters["initial_time"] = t0
            
        # Set time step size
        def set_time_step_size(self, dt):
            assert isinstance(dt, Number)
            self.dt = dt
            self._time_stepping_parameters["time_step_size"] = dt
            
        # Set final time
        def set_final_time(self, T):
            assert isinstance(T, Number)
            self.T = T
            self._time_stepping_parameters["final_time"] = T
            
        # Initialize data structures required for the online phase
        def init(self, current_stage="online"):
            # Initialize first data structures related to initial conditions
            self._init_initial_condition(current_stage)
            self._init_time_series(current_stage)
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
                        initial_condition[component] = OnlineAffineExpansionStorage(Q_ic[component])
                if n_components == 1:
                    self.initial_condition = initial_condition[self.components[0]]
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous[self.components[0]]
                    self.Q_ic = Q_ic[self.components[0]]
                else:
                    self.initial_condition = initial_condition
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous
                    self.Q_ic = Q_ic
                assert self.initial_condition_is_homogeneous == self.truth_problem.initial_condition_is_homogeneous
                # Load initial conditions from file if we are online
                if current_stage == "online":
                    for component in self.components:
                        if not initial_condition_is_homogeneous[component]:
                            self.assemble_operator(initial_condition_string.format(c=component), "online")
                elif current_stage == "offline":
                    pass # Nothing else to be done
                else:
                    raise ValueError("Invalid stage in _init_initial_condition().")
                    
        def _init_time_series(self, current_stage="online"):
            try:
                monitor_t0 = self._time_stepping_parameters["monitor"]["initial_time"]
            except KeyError:
                monitor_t0 = self.t0
            try:
                monitor_dt = self._time_stepping_parameters["monitor"]["time_step_size"]
            except KeyError:
                assert self.dt is not None
                monitor_dt = self.dt
            assert self.T is not None
            monitor_T = self.T
            self._solution_over_time = TimeSeries((monitor_t0, monitor_T), monitor_dt)
            self._solution_dot_over_time = TimeSeries((monitor_t0, monitor_T), monitor_dt)
            self._output_over_time = TimeSeries((monitor_t0, monitor_T), monitor_dt)
            
        # Assemble the reduced order affine expansion.
        def build_reduced_operators(self, current_stage="offline"):
            ParametrizedReducedDifferentialProblem_DerivedClass.build_reduced_operators(self, current_stage)
            # Initial condition
            self._build_reduced_initial_condition(current_stage)
            
        def _build_reduced_initial_condition(self, current_stage="offline"):
            if len(self.components) > 1:
                initial_condition_string = "initial_condition_{c}"
                for component in self.components:
                    if not self.initial_condition_is_homogeneous[component]:
                        self.initial_condition[component] = self.assemble_operator(initial_condition_string.format(c=component), current_stage)
            else:
                if not self.initial_condition_is_homogeneous:
                    self.initial_condition = self.assemble_operator("initial_condition", "offline")
                
        # Assemble the reduced order affine expansion
        def assemble_operator(self, term, current_stage="online"):
            assert current_stage in ("online", "offline")
            if term.startswith("initial_condition"):
                component = term.replace("initial_condition", "").replace("_", "")
                if current_stage == "online": # load from file
                    if component != "":
                        initial_condition = self.initial_condition[component]
                    else:
                        initial_condition = self.initial_condition
                    initial_condition.load(self.folder["reduced_operators"], term)
                elif current_stage == "offline":
                    if component != "":
                        truth_initial_condition = self.truth_problem.initial_condition[component]
                        initial_condition = self.initial_condition[component]
                        truth_projection_inner_product = self.truth_problem.projection_inner_product[component]
                    else:
                        truth_initial_condition = self.truth_problem.initial_condition
                        initial_condition = self.initial_condition
                        truth_projection_inner_product = self.truth_problem.projection_inner_product
                    assert len(truth_projection_inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                    for (q, truth_initial_condition_q) in enumerate(truth_initial_condition):
                        initial_condition[q] = transpose(self.basis_functions)*truth_projection_inner_product[0]*truth_initial_condition_q
                    initial_condition.save(self.folder["reduced_operators"], term)
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
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
                
        def solve(self, N=None, **kwargs):
            N, kwargs = self._online_size_from_kwargs(N, **kwargs)
            N += self.N_bc
            self._latest_solve_kwargs = kwargs
            self._solution = OnlineFunction(N)
            self._solution_over_time.clear()
            self._solution_dot = OnlineFunction(N)
            self._solution_dot_over_time.clear()
            if N == 0: # trivial case
                self._solution_over_time.extend([self._solution for _ in self._solution_over_time.expected_times()])
                self._solution_dot_over_time.extend([self._solution_dot for _ in self._solution_dot_over_time.expected_times()])
                return self._solution_over_time
            try:
                assign(self._solution_over_time, self._solution_over_time_cache[self.mu, N, kwargs]) # **kwargs is not supported by __getitem__
                assign(self._solution_dot_over_time, self._solution_dot_over_time_cache[self.mu, N, kwargs])
            except KeyError:
                assert not hasattr(self, "_is_solving")
                self._is_solving = True
                self._solve(N, **kwargs)
                delattr(self, "_is_solving")
            assign(self._solution, self._solution_over_time[-1])
            assign(self._solution_dot, self._solution_dot_over_time[-1])
            return self._solution_over_time
            
        class ProblemSolver(ParametrizedReducedDifferentialProblem_DerivedClass.ProblemSolver, TimeDependentProblemWrapper):
            def set_time(self, t):
                problem = self.problem
                problem.set_time(t)
                
            def bc_eval(self, t):
                assert self.problem.t == t
                return ParametrizedReducedDifferentialProblem_DerivedClass.ProblemSolver.bc_eval(self)
                
            def ic_eval(self):
                problem = self.problem
                N = self.N
                if len(problem.components) > 1:
                    all_initial_conditions = list()
                    all_initial_conditions_thetas = list()
                    for component in problem.components:
                        if problem.initial_condition[component] and not problem.initial_condition_is_homogeneous[component]:
                            all_initial_conditions.extend(problem.initial_condition[component][:N])
                            all_initial_conditions_thetas.extend(problem.compute_theta("initial_condition_" + component))
                    if len(all_initial_conditions) > 0:
                        all_initial_conditions = tuple(all_initial_conditions)
                        all_initial_conditions = OnlineAffineExpansionStorage(all_initial_conditions)
                        all_initial_conditions_thetas = tuple(all_initial_conditions_thetas)
                    else:
                        all_initial_conditions = None
                        all_initial_conditions_thetas = None
                else:
                    if problem.initial_condition and not problem.initial_condition_is_homogeneous:
                        all_initial_conditions = problem.initial_condition[:N]
                        all_initial_conditions_thetas = problem.compute_theta("initial_condition")
                    else:
                        all_initial_conditions = None
                        all_initial_conditions_thetas = None
                assert (all_initial_conditions is None) == (all_initial_conditions_thetas is None)
                if all_initial_conditions is not None:
                    inner_product_N = problem._combined_projection_inner_product[:N, :N]
                    projected_initial_condition = OnlineFunction(N)
                    solver = OnlineLinearSolver(inner_product_N, projected_initial_condition, sum(product(all_initial_conditions_thetas, all_initial_conditions)))
                    solver.set_parameters(problem._linear_solver_parameters)
                    solver.solve()
                    return projected_initial_condition
                else:
                    return None
                    
            def monitor(self, t, solution, solution_dot):
                problem = self.problem
                solution_copy = copy(solution)
                problem._solution_over_time.append(solution_copy)
                problem._solution_over_time_cache[problem.mu, self.N, self.kwargs].append(solution_copy)
                solution_dot_copy = copy(solution_dot)
                problem._solution_dot_over_time.append(solution_dot_copy)
                problem._solution_dot_over_time_cache[problem.mu, self.N, self.kwargs].append(solution_dot_copy)
                
            def solve(self):
                problem = self.problem
                assert len(problem._solution_over_time) == 0
                problem._solution_over_time_cache[problem.mu, self.N, self.kwargs] = copy(problem._solution_over_time)
                assert len(problem._solution_dot_over_time) == 0
                problem._solution_dot_over_time_cache[problem.mu, self.N, self.kwargs] = copy(problem._solution_dot_over_time)
                solver = OnlineTimeStepping(self, problem._solution, problem._solution_dot)
                solver.set_parameters(problem._time_stepping_parameters)
                solver.solve()
                
        # Perform an online evaluation of the output
        def compute_output(self):
            N = self._solution.N
            kwargs = self._latest_solve_kwargs
            try:
                assign(self._output_over_time, self._output_over_time_cache[self.mu, N, kwargs]) # **kwargs is not supported by __getitem__
            except KeyError:
                try:
                    self._compute_output(N)
                except ValueError: # raised by compute_theta if output computation is optional
                    self._output_over_time.clear()
                    self._output_over_time.extend([NotImplemented]*len(self._solution_over_time))
                    self._output = NotImplemented
                self._output_over_time_cache[self.mu, N, kwargs] = self._output_over_time
            else:
                self._output = self._output_over_time[-1]
            return self._output_over_time
            
        # Perform an online evaluation of the output. Internal method
        def _compute_output(self, N):
            self._output_over_time.clear()
            self._output_over_time.extend([NotImplemented]*len(self._solution_over_time))
            self._output = NotImplemented
            
        def _lifting_truth_solve(self, term, i):
            assert term.startswith("dirichlet_bc")
            component = term.replace("dirichlet_bc", "").replace("_", "")
            # Since lifting solves for different values of i are associated to the same parameter
            # but with a patched call to compute_theta(), which returns the i-th component, we set
            # a custom cache_key so that they are properly differentiated when reading from cache.
            lifting_over_time = self.truth_problem.solve(cache_key="lifting_" + component + "_" + str(i))
            times = lifting_over_time.stored_times()
            theta_over_time = list()
            for t in times:
                self.truth_problem.set_time(t)
                theta_over_time.append(self.truth_problem.compute_theta(term)[i])
            # We average the time dependent solution to be used as time independent lifting.
            # Do not even bother adding the initial condition if it is zero
            if not isclose(times[0], self.truth_problem.t0, self.truth_problem.dt/2):
                has_non_homogeneous_initial_condition = True
            else:
                if component != "":
                    assert component in self.truth_problem.components
                    has_non_homogeneous_initial_condition = self.truth_problem.initial_condition[component] and not self.truth_problem.initial_condition_is_homogeneous[component]
                else:
                    assert len(self.truth_problem.components) == 1
                    component = None
                    has_non_homogeneous_initial_condition = self.truth_problem.initial_condition and not self.truth_problem.initial_condition_is_homogeneous
            if has_non_homogeneous_initial_condition:
                time_interval = (times[0], times[-1])
            else:
                time_interval = (times[1], times[-1])
                lifting_over_time = lifting_over_time[1:]
                theta_over_time = theta_over_time[1:]
            # Compute the average and return
            lifting_quadrature = TimeQuadrature(time_interval, lifting_over_time)
            theta_quadrature = TimeQuadrature(time_interval, theta_over_time)
            lifting = lifting_quadrature.integrate()
            lifting /= theta_quadrature.integrate()
            return lifting
            
        def project(self, snapshot_over_time, N=None, on_dirichlet_bc=True, **kwargs):
            projected_snapshot_N_over_time = TimeSeries(snapshot_over_time)
            for snapshot in snapshot_over_time:
                projected_snapshot_N = ParametrizedReducedDifferentialProblem_DerivedClass.project(self, snapshot, N, on_dirichlet_bc, **kwargs)
                projected_snapshot_N_over_time.append(projected_snapshot_N)
            return projected_snapshot_N_over_time
            
        # Internal method for error computation
        def _compute_error(self, **kwargs):
            error_over_time = TimeSeries(self._solution_over_time)
            assert len(self.truth_problem._solution_over_time) == len(self._solution_over_time)
            for (k, t) in enumerate(self.truth_problem._solution_over_time.stored_times()):
                self.set_time(t)
                assign(self._solution, self._solution_over_time[k])
                assign(self.truth_problem._solution, self.truth_problem._solution_over_time[k])
                error = ParametrizedReducedDifferentialProblem_DerivedClass._compute_error(self, **kwargs)
                error_over_time.append(error)
            error_over_time = self._convert_error_over_time(error_over_time)
            return error_over_time
            
        # Internal method for relative error computation
        def _compute_relative_error(self, absolute_error_over_time, **kwargs):
            relative_error_over_time = TimeSeries(self._solution_over_time)
            assert len(self._solution_over_time) == len(absolute_error_over_time)
            for (k, t) in enumerate(self.truth_problem._solution_over_time.stored_times()):
                self.set_time(t)
                assign(self.truth_problem._solution, self.truth_problem._solution_over_time[k])
                absolute_error = self._convert_error_at_time(k, absolute_error_over_time)
                relative_error = ParametrizedReducedDifferentialProblem_DerivedClass._compute_relative_error(self, absolute_error, **kwargs)
                relative_error_over_time.append(relative_error)
            relative_error_over_time = self._convert_error_over_time(relative_error_over_time)
            return relative_error_over_time
            
        # Internal method for output error computation
        def _compute_error_output(self, **kwargs):
            error_output_over_time = TimeSeries(self._output_over_time)
            assert len(self.truth_problem._output_over_time) == len(self._output_over_time)
            for (k, t) in enumerate(self.truth_problem._output_over_time.stored_times()):
                self.set_time(t)
                self._output = self._output_over_time[k]
                self.truth_problem._output = self.truth_problem._output_over_time[k]
                error_output = ParametrizedReducedDifferentialProblem_DerivedClass._compute_error_output(self, **kwargs)
                error_output_over_time.append(error_output)
            return error_output_over_time
            
        # Internal method for output relative error computation
        def _compute_relative_error_output(self, absolute_error_output_over_time, **kwargs):
            relative_error_output_over_time = TimeSeries(self._output_over_time)
            assert len(self.truth_problem._output_over_time) == len(absolute_error_output_over_time)
            for (k, t) in enumerate(self.truth_problem._output_over_time.stored_times()):
                self.set_time(t)
                self.truth_problem._output = self.truth_problem._output_over_time[k]
                relative_error_output = ParametrizedReducedDifferentialProblem_DerivedClass._compute_relative_error_output(self, absolute_error_output_over_time[k], **kwargs)
                relative_error_output_over_time.append(relative_error_output)
            return relative_error_output_over_time
        
        def _convert_error_over_time(self, error_over_time):
            """
            This internal method converts the error over time as follows:
             - if the problem has only one component, no conversion is required;
             - if the problem has more than one component, then a conversion is preformed such that
               the result is a dict from components to error over time for that component, rather than
               a list (over time) of dicts (over components)
            """
            assert isinstance(error_over_time[0], (dict, Number))
            if isinstance(error_over_time[0], dict):
                assert all([isinstance(error, dict) for error in error_over_time])
                components = list(error_over_time[0].keys())
                assert all([list(error.keys()) == components for error in error_over_time])
                output = dict()
                for component in components:
                    output[component] = TimeSeries(error_over_time)
                    for error in error_over_time:
                        output[component].append(error[component])
                return output
            else:
                assert all([isinstance(error, Number) for error in error_over_time])
                return error_over_time
        
        def _convert_error_at_time(self, k, converted_error_over_time):
            """
            This internal method converts back the error at time step k as follows:
             - if the problem has only one component, no conversion is required, and the error at time
               k is returned by extracting the k-th index of the input;
             - if the problem has more than one component, then a conversion is performed such that
               the result is a dict from components to error at time step k
            """
            assert (
                isinstance(converted_error_over_time, dict)
                    or
                hasattr(converted_error_over_time, "__iter__") # isinstance(converted_error_over_time, TimeSeries)
            )
            if isinstance(converted_error_over_time, dict):
                output = dict()
                for (component, error_over_time_for_component) in converted_error_over_time.items():
                    assert all([isinstance(error, Number) for error in error_over_time_for_component])
                    output[component] = error_over_time_for_component[k]
                return output
            else:
                assert all([isinstance(error, Number) for error in converted_error_over_time])
                return converted_error_over_time[k]
        
        # Export solution to file
        def export_solution(self, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            for (k, solution) in enumerate(solution_over_time):
                N = solution.N
                assert suffix is None
                self.truth_problem.export_solution(folder, filename, self.basis_functions[:N]*solution, component=component, suffix=k)
            
        def export_error(self, folder=None, filename=None, component=None, suffix=None, **kwargs):
            self.truth_problem.solve(**kwargs)
            assert len(self.truth_problem._solution_over_time) == len(self._solution_over_time)
            for (k, t) in enumerate(self.truth_problem._solution_over_time.stored_times()):
                self.set_time(t)
                assign(self._solution, self._solution_over_time[k])
                assign(self.truth_problem._solution, self.truth_problem._solution_over_time[k])
                assert suffix is None
                self.truth_problem.export_solution(folder, filename, self.truth_problem._solution - self.basis_functions[:self._solution.N]*self._solution, component=component, suffix=k)
            
        def export_output(self, folder=None, filename=None, output_over_time=None, suffix=None):
            self.truth_problem.export_output(folder, filename, output_over_time, suffix)
            
    # return value (a class) for the decorator
    return TimeDependentReducedProblem_Class
