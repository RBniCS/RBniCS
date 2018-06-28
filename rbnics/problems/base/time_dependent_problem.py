# Copyright (C) 2015-2018 by the RBniCS authors
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
from numpy import arange
from rbnics.backends import AffineExpansionStorage, assign, copy, Function, product, sum, TimeDependentProblem1Wrapper, TimeSeries, TimeStepping
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators
from rbnics.utils.test import PatchInstanceMethod

@RequiredBaseDecorators(None)
def TimeDependentProblem(ParametrizedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class TimeDependentProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        # Default initialization of members
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
            # Store quantities related to the time discretization
            self.t = 0.
            self.t0 = 0.
            self.dt = None
            self.T = None
            # Additional options for time stepping may be stored in the following dict
            self._time_stepping_parameters = dict()
            self._time_stepping_parameters["initial_time"] = self.t0
            # Matrices/vectors resulting from the truth discretization
            self.initial_condition = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components)
            self.initial_condition_is_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
            # Time derivative of the solution, at the current time
            self._solution_dot = Function(self.V)
            # Solution and output over time
            self._solution_over_time = None # TimeSeries of Functions
            self._solution_dot_over_time = None # TimeSeries of Functions
            self._output_over_time = None # TimeSeries of numbers
            # I/O
            def _solution_cache_key_generator(*args, **kwargs):
                assert len(args) is 1
                assert args[0] == self.mu
                return self._cache_key_from_kwargs(**kwargs)
            def _solution_cache_import(filename):
                self.import_solution(self.folder["cache"], filename, self._solution_over_time)
                return self._solution_over_time
            def _solution_cache_export(filename):
                self.export_solution(self.folder["cache"], filename, self._solution_over_time)
            def _solution_cache_filename_generator(*args, **kwargs):
                assert len(args) is 1
                assert args[0] == self.mu
                return self._cache_file_from_kwargs(**kwargs)
            self._solution_over_time_cache = Cache(
                "problems",
                key_generator=_solution_cache_key_generator,
                import_=_solution_cache_import,
                export=_solution_cache_export,
                filename_generator=_solution_cache_filename_generator
            )
            def _solution_dot_cache_key_generator(*args, **kwargs):
                assert len(args) is 1
                assert args[0] == self.mu
                return self._cache_key_from_kwargs(**kwargs)
            def _solution_dot_cache_import(filename):
                self.import_solution(self.folder["cache"], filename + "_dot", self._solution_dot_over_time)
                return self._solution_over_time
            def _solution_dot_cache_export(filename):
                self.export_solution(self.folder["cache"], filename + "_dot", self._solution_dot_over_time)
            def _solution_dot_cache_filename_generator(*args, **kwargs):
                assert len(args) is 1
                assert args[0] == self.mu
                return self._cache_file_from_kwargs(**kwargs)
            self._solution_dot_over_time_cache = Cache(
                "problems",
                key_generator=_solution_dot_cache_key_generator,
                import_=_solution_dot_cache_import,
                export=_solution_dot_cache_export,
                filename_generator=_solution_dot_cache_filename_generator
            )
            del self._solution_cache
            def _output_cache_key_generator(*args, **kwargs):
                assert len(args) is 1
                assert args[0] == self.mu
                return self._cache_key_from_kwargs(**kwargs)
            def _output_cache_import(filename):
                self.import_output(self.folder["cache"], filename)
                return self._output_over_time
            def _output_cache_export(filename):
                self.export_output(self.folder["cache"], filename)
            def _output_cache_filename_generator(*args, **kwargs):
                assert len(args) is 1
                assert args[0] == self.mu
                return self._cache_file_from_kwargs(**kwargs)
            self._output_over_time_cache = Cache(
                "problems",
                key_generator=_output_cache_key_generator,
                import_=_output_cache_import,
                export=_output_cache_export,
                filename_generator=_output_cache_filename_generator
            )
            del self._output_cache
            
        # Set current time
        def set_time(self, t):
            assert isinstance(t, Number)
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
            
        # Export solution to file
        def export_solution(self, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
            if folder is None:
                folder = self.folder_prefix
            if filename is None:
                filename = "solution"
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            assert suffix is None
            for (k, solution) in enumerate(solution_over_time):
                ParametrizedDifferentialProblem_DerivedClass.export_solution(self, folder, filename, solution, component=component, suffix=k)
                
        # Import solution from file
        def import_solution(self, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
            if folder is None:
                folder = self.folder_prefix
            if filename is None:
                filename = "solution"
            solution = Function(self.V)
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            assert suffix is None
            k = 0
            all_t = arange(self.t0, self.T + self.dt/2., self.dt).tolist()
            del solution_over_time[:]
            for self.t in all_t:
                ParametrizedDifferentialProblem_DerivedClass.import_solution(self, folder, filename, solution, component, suffix=k)
                solution_over_time.append(copy(solution))
                k += 1
                
        def export_output(self, folder=None, filename=None, output_over_time=None, suffix=None):
            if folder is None:
                folder = self.folder_prefix
            if filename is None:
                filename = "solution"
            if output_over_time is None:
                output_over_time = self._output_over_time
            assert suffix is None
            for (k, output) in enumerate(output_over_time):
                ParametrizedDifferentialProblem_DerivedClass.export_output(self, folder, filename, [output], suffix=k)
                
        def import_output(self, folder=None, filename=None, output_over_time=None, suffix=None):
            if folder is None:
                folder = self.folder_prefix
            if filename is None:
                filename = "solution"
            output = [0.]
            if output_over_time is None:
                output_over_time = self._output_over_time
            assert suffix is None
            k = 0
            all_t = arange(self.t0, self.T + self.dt/2., self.dt).tolist()
            del output_over_time[:]
            for self.t in all_t:
                ParametrizedDifferentialProblem_DerivedClass.import_output(self, folder, filename, output, suffix=k)
                assert len(output) is 1
                output_over_time.append(output[0])
                k += 1
                
        # Initialize data structures required for the offline phase
        def init(self):
            ParametrizedDifferentialProblem_DerivedClass.init(self)
            self._init_initial_condition()
            self._init_time_series()
            
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
                    self.initial_condition = initial_condition[self.components[0]]
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous[self.components[0]]
                else:
                    self.initial_condition = initial_condition
                    self.initial_condition_is_homogeneous = initial_condition_is_homogeneous
                # We enforce consistency between Dirichlet BCs and IC, as in the following cases:
                # a) (homogeneous Dirichlet BCs, homogeneous IC): nothing to be enforced
                # b) (non homogeneous Dirichlet BCs, homogeneous IC): we enforce that the theta
                #     term of each Dirichlet BC is zero at t = 0, resulting in homogeneous Dirichlet BCs
                #     at t = 0. This is needed e.g. in order to make sure that while post processing a snapshot
                #     subtracting the lifting at t = 0 doesn't change the solution (which must remain zero)
                # c) (homogeneous Dirichlet BCs, non homogeneous IC): we trust that the non homogeneous IC
                #     provided by the user is actually zero on the Dirichlet boundaries, otherwise there will
                #     be no way to accurately recover it by projecting on a space which only has bases equal
                #     to zero on the Dirichlet boundary.
                # d) (non homogeneous Dirichlet BCs, non homogeneous IC): we trust that the restriction of IC
                #     on the Dirichlet boundary is equal to the evaluation of the Dirichlet BCs at t = 0.
                #     If that were not true, than post processing a snapshot subtracting the lifting at t = 0
                #     would result in a postprocessed snapshot which is not zero on the Dirichlet boundary, thus
                #     adding an element with non zero value on the Dirichlet boundary to the basis
                for component in self.components:
                    if len(self.components) > 1:
                        has_homogeneous_dirichlet_bc = self.dirichlet_bc_are_homogeneous[component]
                        has_homogeneous_initial_condition = self.initial_condition_is_homogeneous[component]
                        dirichlet_bc_string = "dirichlet_bc_{c}"
                    else:
                        has_homogeneous_dirichlet_bc = self.dirichlet_bc_are_homogeneous
                        has_homogeneous_initial_condition = self.initial_condition_is_homogeneous
                        dirichlet_bc_string = "dirichlet_bc"
                    if has_homogeneous_dirichlet_bc and has_homogeneous_initial_condition: # case a)
                        pass
                    elif not has_homogeneous_dirichlet_bc and has_homogeneous_initial_condition: # case b)
                        def generate_modified_compute_theta(component):
                            standard_compute_theta = self.compute_theta
                            def modified_compute_theta(self_, term):
                                if term == dirichlet_bc_string.format(c=component):
                                    theta_bc = standard_compute_theta(term)
                                    if self_.t == 0.:
                                        return (0.,)*len(theta_bc)
                                    else:
                                        return theta_bc
                                else:
                                    return standard_compute_theta(term)
                            return modified_compute_theta
                        PatchInstanceMethod(self, "compute_theta", generate_modified_compute_theta(component)).patch()
                    elif has_homogeneous_dirichlet_bc and not has_homogeneous_initial_condition: # case c)
                        pass
                    elif not has_homogeneous_dirichlet_bc and not has_homogeneous_initial_condition: # case d)
                        pass
                    else:
                        raise RuntimeError("Impossible to arrive here.")
                        
        def _init_time_series(self):
            assert self.dt is not None
            assert self.T is not None
            self._solution_over_time = TimeSeries((self.t0, self.T), self.dt)
            self._solution_dot_over_time = TimeSeries((self.t0, self.T), self.dt)
            self._output_over_time = TimeSeries((self.t0, self.T), self.dt)
        
        def solve(self, **kwargs):
            self._latest_solve_kwargs = kwargs
            try:
                assign(self._solution_over_time, self._solution_over_time_cache[self.mu, kwargs]) # **kwargs is not supported by __getitem__
                assign(self._solution_dot_over_time, self._solution_dot_over_time_cache[self.mu, kwargs])
            except KeyError:
                assert not hasattr(self, "_is_solving")
                self._is_solving = True
                assign(self._solution, Function(self.V))
                assign(self._solution_dot, Function(self.V))
                self._solve(**kwargs)
                delattr(self, "_is_solving")
                self._solution_over_time_cache[self.mu, kwargs] = copy(self._solution_over_time)
                self._solution_dot_over_time_cache[self.mu, kwargs] = copy(self._solution_dot_over_time)
            else:
                assign(self._solution, self._solution_over_time[-1])
                assign(self._solution_dot, self._solution_dot_over_time[-1])
            return self._solution_over_time
            
        class ProblemSolver(ParametrizedDifferentialProblem_DerivedClass.ProblemSolver, TimeDependentProblem1Wrapper):
            def set_time(self, t):
                problem = self.problem
                problem.set_time(t)
                
            def bc_eval(self, t):
                assert self.problem.t == t
                return ParametrizedDifferentialProblem_DerivedClass.ProblemSolver.bc_eval(self)
                
            def ic_eval(self):
                problem = self.problem
                if len(problem.components) > 1:
                    all_initial_conditions = list()
                    all_initial_conditions_thetas = list()
                    for component in problem.components:
                        if problem.initial_condition[component] is not None:
                            all_initial_conditions.extend(problem.initial_condition[component])
                            all_initial_conditions_thetas.extend(problem.compute_theta("initial_condition_" + component))
                    if len(all_initial_conditions) > 0:
                        all_initial_conditions = tuple(all_initial_conditions)
                        all_initial_conditions = AffineExpansionStorage(all_initial_conditions)
                        all_initial_conditions_thetas = tuple(all_initial_conditions_thetas)
                    else:
                        all_initial_conditions = None
                        all_initial_conditions_thetas = None
                else:
                    if problem.initial_condition is not None:
                        all_initial_conditions = problem.initial_condition
                        all_initial_conditions_thetas = problem.compute_theta("initial_condition")
                    else:
                        all_initial_conditions = None
                        all_initial_conditions_thetas = None
                assert (all_initial_conditions is None) == (all_initial_conditions_thetas is None)
                if all_initial_conditions is not None:
                    return sum(product(all_initial_conditions_thetas, all_initial_conditions))
                else:
                    return Function(problem.V)
                    
            def solve(self):
                problem = self.problem
                solver = TimeStepping(self, problem._solution, problem._solution_dot)
                solver.set_parameters(problem._time_stepping_parameters)
                (_, problem._solution_over_time, problem._solution_dot_over_time) = solver.solve()
                assign(problem._solution, problem._solution_over_time[-1])
                assign(problem._solution_dot, problem._solution_dot_over_time[-1])
        
        # Perform a truth evaluation of the output
        def compute_output(self):
            """
            
            :return: output evaluation.
            """
            kwargs = self._latest_solve_kwargs
            try:
                assign(self._output_over_time, self._output_over_time_cache[self.mu, kwargs]) # **kwargs is not supported by __getitem__
            except KeyError:
                try:
                    self._compute_output()
                except ValueError: # raised by compute_theta if output computation is optional
                    del self._output_over_time[:]
                    self._output_over_time.extend([NotImplemented]*len(self._solution_over_time))
                    self._output = NotImplemented
                self._output_over_time_cache[self.mu, kwargs] = self._output_over_time
            else:
                self._output = self._output_over_time[-1]
            return self._output_over_time
            
        # Perform a truth evaluation of the output
        def _compute_output(self):
            del self._output_over_time[:]
            self._output_over_time.extend([NotImplemented]*len(self._solution_over_time))
            self._output = NotImplemented
            
    # return value (a class) for the decorator
    return TimeDependentProblem_Class
