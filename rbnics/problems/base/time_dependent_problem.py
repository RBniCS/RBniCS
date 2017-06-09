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

from rbnics.backends import AffineExpansionStorage, assign, copy, Function, TimeDependentProblem1Wrapper, TimeStepping
from rbnics.utils.decorators import apply_decorator_only_once, Extends, override
from rbnics.utils.mpi import log, PROGRESS

@apply_decorator_only_once
def TimeDependentProblem(ParametrizedDifferentialProblem_DerivedClass):
    
    TimeDependentProblem_Base = ParametrizedDifferentialProblem_DerivedClass
    
    @Extends(TimeDependentProblem_Base, preserve_class_name=True)
    class TimeDependentProblem_Class(TimeDependentProblem_Base):
        ## Default initialization of members
        @override
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            TimeDependentProblem_Base.__init__(self, V, **kwargs)
            # Store quantities related to the time discretization
            self.t = 0.
            self.t0 = 0.
            self.dt = None
            self.T  = None
            # Additional options for time stepping may be stored in the following dict
            self._time_stepping_parameters = dict()
            self._time_stepping_parameters["initial_time"] = self.t0
            # Matrices/vectors resulting from the truth discretization
            self.initial_condition = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components)
            self.initial_condition_is_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
            # Time derivative of the solution, at the current time
            self._solution_dot = Function(self.V)
            self._solution_dot_cache = dict() # of Functions
            # Solution and output over time
            self._solution_over_time = list() # of Functions
            self._solution_dot_over_time = list() # of Functions
            self._solution_over_time_cache = dict() # of list of Functions
            self._solution_dot_over_time_cache = dict() # of list of Functions
            self._output_over_time = list() # of floats
            self._output_over_time_cache = dict() # of list of floats

        ## Set current time
        def set_time(self, t):
            assert isinstance(t, (float, int))
            t = float(t)
            self.t = t
            
        ## Set initial time
        def set_initial_time(self, t0):
            assert isinstance(t0, (float, int))
            t0 = float(t0)
            self.t0 = t0
            self._time_stepping_parameters["initial_time"] = t0
                    
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
        def export_solution(self, folder, filename, solution_over_time=None, solution_dot_over_time=None, component=None, suffix=None):
            if solution_over_time is None:
                solution_over_time = self._solution_over_time
            if solution_dot_over_time is None:
                solution_dot_over_time = self._solution_dot_over_time
            assert suffix is None
            for (k, (solution, solution_dot)) in enumerate(zip(solution_over_time, solution_dot_over_time)):
                TimeDependentProblem_Base.export_solution(self, folder + "/" + filename, "solution", solution, component=component, suffix=k)
                TimeDependentProblem_Base.export_solution(self, folder + "/" + filename, "solution_dot", solution_dot, component=component, suffix=k)
                
        ## Import solution from file
        @override
        def import_solution(self, folder, filename, solution_over_time=None, solution_dot_over_time=None, suffix=None):
            if solution_over_time is None:
                solution = self._solution
                solution_over_time = self._solution_over_time
            else:
                solution = Function(self.V)
            if solution_dot_over_time is None:
                solution_dot = self._solution_dot
                solution_dot_over_time = self._solution_dot_over_time
            else:
                solution_dot = Function(self.V)
            assert suffix is None
            k = 0
            self.t = 0
            self._time_stepping_parameters["initial_time"] = 0.
            del solution_over_time[:]
            del solution_dot_over_time[:]
            while self.t <= self.T:
                import_solution = TimeDependentProblem_Base.import_solution(self, folder + "/" + filename, "solution", solution, suffix=k)
                import_solution_dot = TimeDependentProblem_Base.import_solution(self, folder + "/" + filename, "solution_dot", solution_dot, suffix=k)
                import_solution_and_solution_dot = import_solution and import_solution_dot
                if import_solution_and_solution_dot:
                    solution_over_time.append(copy(self._solution))
                    solution_dot_over_time.append(copy(self._solution_dot))
                    k += 1
                    self.t += self.dt
                else:
                    if k > 0:
                        k -= 1
                        self.t -= self.dt
                        assign(solution, solution_over_time[k])
                        assign(solution_dot, solution_dot_over_time[k])
                        self._time_stepping_parameters["initial_time"] = self.t
                    return False
            return True
                
        ## Initialize data structures required for the offline phase
        @override
        def init(self):
            TimeDependentProblem_Base.init(self)
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
        
        @override
        def solve(self, **kwargs):
            (cache_key, cache_file) = self._cache_key_and_file_from_kwargs(**kwargs)
            assert (
                (cache_key in self._solution_cache)
                    ==
                (cache_key in self._solution_dot_cache)
                    ==
                (cache_key in self._solution_over_time_cache)
                    ==
                (cache_key in self._solution_dot_over_time_cache)
            )
            if cache_key in self._solution_cache:
                log(PROGRESS, "Loading truth solution from cache")
                assign(self._solution, self._solution_cache[cache_key])
                assign(self._solution_dot, self._solution_dot_cache[cache_key])
                assign(self._solution_over_time, self._solution_over_time_cache[cache_key])
                assign(self._solution_dot_over_time, self._solution_dot_over_time_cache[cache_key])
            elif self.import_solution(self.folder["cache"], cache_file):
                log(PROGRESS, "Loading truth solution from file")
                self._solution_cache[cache_key] = copy(self._solution)
                self._solution_dot_cache[cache_key] = copy(self._solution_dot)
                self._solution_over_time_cache[cache_key] = copy(self._solution_over_time)
                self._solution_dot_over_time_cache[cache_key] = copy(self._solution_dot_over_time)
            else:
                log(PROGRESS, "Solving truth problem")
                assert not hasattr(self, "_is_solving")
                self._is_solving = True
                assign(self._solution, Function(self.V))
                assign(self._solution_dot, Function(self.V))
                self._solve(**kwargs)
                delattr(self, "_is_solving")
                self._solution_cache[cache_key] = copy(self._solution)
                self._solution_dot_cache[cache_key] = copy(self._solution_dot)
                self._solution_over_time_cache[cache_key] = copy(self._solution_over_time)
                self._solution_dot_over_time_cache[cache_key] = copy(self._solution_dot_over_time)
                self.export_solution(self.folder["cache"], cache_file)
            return self._solution_over_time
            
        class ProblemSolver(TimeDependentProblem_Base.ProblemSolver, TimeDependentProblem1Wrapper):
            def bc_eval(self, t):
                problem = self.problem
                problem.set_time(t)
                return TimeDependentProblem_Base.ProblemSolver.bc_eval(self)
                
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
                solver = TimeStepping(self, problem._solution)
                solver.set_parameters(problem._time_stepping_parameters)
                (_, problem._solution_over_time, problem._solution_dot_over_time) = solver.solve()
                assign(problem._solution, problem._solution_over_time[-1])
                assign(problem._solution_dot, problem._solution_dot_over_time[-1])
        
        ## Perform a truth evaluation of the output
        @override
        def compute_output(self):
            """
            
            :return: output evaluation.
            """
            cache_key = self._output_cache__current_cache_key
            assert (
                (cache_key in self._output_cache)
                    ==
                (cache_key in self._output_over_time_cache)
            )
            if cache_key in self._output_cache:
                log(PROGRESS, "Loading truth output from cache")
                self._output = self._output_cache[cache_key]
                self._output_over_time = self._output_over_time_cache[cache_key]
            else: # No precomputed output available. Truth output is performed.
                log(PROGRESS, "Computing truth output")
                self._compute_output()
                self._output_cache[cache_key] = self._output
                self._output_over_time_cache[cache_key] = self._output_over_time
            return self._output_over_time
            
        ## Perform a truth evaluation of the output
        @override
        def _compute_output(self):
            self._output_over_time = [NotImplemented]*len(self._solution_over_time)
            self._output = NotImplemented
            
    # return value (a class) for the decorator
    return TimeDependentProblem_Class
    
