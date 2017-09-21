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


from numpy import arange, asarray, linspace
try:
    from assimulo.solvers import IDA
    from assimulo.solvers.sundials import IDAError
    from assimulo.problem import Implicit_Problem
except ImportError:
    has_IDA = False
else:
    has_IDA = True
from rbnics.backends.abstract import TimeStepping as AbstractTimeStepping, TimeDependentProblemWrapper
from rbnics.backends.online.basic.wrapping import DirichletBC
from rbnics.backends.online.numpy.assign import assign
from rbnics.backends.online.numpy.copy import function_copy
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.linear_solver import LinearSolver
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.nonlinear_solver import NonlinearSolver, NonlinearProblemWrapper
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import BackendFor

@BackendFor("numpy", inputs=(TimeDependentProblemWrapper, Function.Type(), Function.Type(), (Function.Type(), None)))
class TimeStepping(AbstractTimeStepping):
    def __init__(self, problem_wrapper, solution, solution_dot, solution_dot_dot=None):
        assert problem_wrapper.time_order() in (1, 2)
        if problem_wrapper.time_order() == 1:
            assert solution_dot_dot is None
            ic = problem_wrapper.ic_eval()
            if ic is not None:
                assign(solution, ic)
            self.problem = _TimeDependentProblem1(problem_wrapper.residual_eval, solution, solution_dot, problem_wrapper.bc_eval, problem_wrapper.jacobian_eval, problem_wrapper.set_time)
            self.solver  = self.problem.create_solver({"problem_type": "linear"})
        elif problem_wrapper.time_order() == 2:
            assert solution_dot_dot is not None
            ic_eval_output = problem_wrapper.ic_eval()
            assert isinstance(ic_eval_output, tuple) or ic_eval_output is None
            if ic_eval_output is not None:
                assert len(ic_eval_output) == 2
                assign(solution, ic_eval_output[0])
                assign(solution_dot, ic_eval_output[1])
            self.problem = _TimeDependentProblem2(problem_wrapper.residual_eval, solution, solution_dot, solution_dot_dot, problem_wrapper.bc_eval, problem_wrapper.jacobian_eval, problem_wrapper.set_time)
            self.solver  = self.problem.create_solver({"problem_type": "nonlinear"})
        else:
            raise ValueError("Invalid time order in TimeStepping.__init__().")
                        
    def set_parameters(self, parameters):
        self.solver = self.problem.create_solver(parameters)
                
    def solve(self):
        return self.solver.solve()
        
class _TimeDependentProblem1(object):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_dot = solution_dot
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        
    def create_solver(self, parameters=None):
        if parameters is None:
            parameters = {}
        if "integrator_type" not in parameters:
            integrator_type = "beuler"
        else:
            integrator_type = parameters["integrator_type"]
        if "problem_type" not in parameters:
            problem_type = "linear"
        else:
            problem_type = parameters["problem_type"]
        
        if has_IDA:
            assert integrator_type in ("beuler", "ida")
        else:
            assert integrator_type in ("beuler")
        assert problem_type in ("linear", "nonlinear")
        
        if integrator_type == "beuler":
            solver = _ScipyImplicitEuler(self.residual_eval, self.solution, self.solution_dot, self.bc_eval, self.jacobian_eval, self.set_time, problem_type)
            solver.set_parameters(parameters)
            return solver
        elif integrator_type == "ida":
            solver = _AssimuloIDA(self.residual_eval, self.solution, self.solution_dot, self.bc_eval, self.jacobian_eval, self.set_time)
            solver.set_parameters(parameters)
            return solver
        else:
            raise ValueError("Invalid integrator type in _TimeDependentProblem_Base.create_solver().")
        
class _ScipyImplicitEuler(object):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time, problem_type):
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_dot = solution_dot
        self.solution_previous = Function(solution.vector().N) # equal to zero
        self.zero = Function(self.solution.vector().N) # equal to zero
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        self.problem_type = problem_type
        # Setup solver
        if problem_type == "linear":
            class _LinearSolver(LinearSolver):
                def __init__(self_, t):
                    self.set_time(t)
                    minus_solution_previous_over_dt = self.solution_previous
                    minus_solution_previous_over_dt.vector()[:] /= - self._time_step_size
                    lhs = self.jacobian_eval(t, self.zero, self.zero, 1./self._time_step_size)
                    rhs = - self.residual_eval(t, self.zero, minus_solution_previous_over_dt)
                    bcs_t = self.bc_eval(t)
                    LinearSolver.__init__(self_, lhs, self.solution, rhs, bcs_t)
                
            self.solver_generator = _LinearSolver
        elif problem_type == "nonlinear":
            class _NonlinearSolver(NonlinearSolver):
                def __init__(self_, t):
                    class _NonlinearProblemWrapper(NonlinearProblemWrapper):
                        def __init__(self_):
                            self.set_time(t)
                        def _store_solution_and_solution_dot(self_, solution):
                            self.solution.vector()[:] = solution.vector()
                            self.solution_dot.vector()[:] = (solution.vector() - self.solution_previous.vector())/self._time_step_size
                        def jacobian_eval(self_, solution):
                            self_._store_solution_and_solution_dot(solution)
                            return self.jacobian_eval(t, self.solution, self.solution_dot, 1./self._time_step_size)
                        def residual_eval(self_, solution):
                            self_._store_solution_and_solution_dot(solution)
                            return self.residual_eval(t, self.solution, self.solution_dot)
                        def bc_eval(self_):
                            return self.bc_eval(t)
                    NonlinearSolver.__init__(self_, _NonlinearProblemWrapper(), self.solution)
                
            self.solver_generator = _NonlinearSolver
        # Additional storage which will be setup by set_parameters
        self._final_time = None
        self._initial_time = 0.
        self._nonlinear_solver_parameters = None
        self._max_time_steps = None
        self._monitor = None
        self._report = None
        self._time_step_size = None
    
    def set_parameters(self, parameters):
        for (key, value) in parameters.items():
            if key == "final_time":
                self._final_time = value
            elif key == "initial_time":
                self._initial_time = value
            elif key == "integrator_type":
                assert value == "beuler"
            elif key == "max_time_steps":
                self._max_time_steps = value
            elif key == "monitor":
                self._monitor = value
            elif key == "nonlinear_solver":
                self._nonlinear_solver_parameters = value
            elif key == "problem_type":
                assert value == self.problem_type
            elif key == "report":
                if value == True:
                    def print_time(t):
                        print("# t = " + str(t))
                    self._report = print_time
                else:
                    self._report = None
            elif key == "time_step_size":
                self._time_step_size = value
            else:
                raise ValueError("Invalid paramater passed to _ScipyImplicitEuler object.")
                
    def solve(self):
        assert self._max_time_steps is not None or self._time_step_size is not None
        if self._time_step_size is not None:
            all_t = arange(self._initial_time, self._final_time + self._time_step_size, self._time_step_size)
        elif self._max_time_steps is not None:
            all_t = linspace(self._initial_time, self._final_time, num=self._max_time_steps+1)
            self._time_step_size = float(all_t[2] - all_t[1])
            
        all_solutions = list()
        all_solutions.append(function_copy(self.solution))
        all_solutions_dot = list()
        all_solutions_dot.append(function_copy(self.solution_dot))
        self.solution_previous.vector()[:] = self.solution.vector()
        for t in all_t[1:]:
            if self._report is not None:
                self._report(t)
            solver = self.solver_generator(t)
            if self.problem_type == "nonlinear":
                if self._nonlinear_solver_parameters is not None:
                    solver.set_parameters(self._nonlinear_solver_parameters)
            solver.solve()
            all_solutions.append(function_copy(self.solution))
            self.solution_dot.vector()[:] = (all_solutions[-1].vector() - all_solutions[-2].vector())/self._time_step_size
            all_solutions_dot.append(function_copy(self.solution_dot))
            self.solution_previous.vector()[:] = self.solution.vector()
            if self._monitor is not None:
                self._monitor(t, self.solution, self.solution_dot)
        
        return all_t, all_solutions, all_solutions_dot
        
if has_IDA:
    class _AssimuloIDA(object):
        def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
            self.residual_eval = residual_eval
            self.solution = solution
            self.solution_dot = solution_dot
            self.bc_eval = bc_eval
            self.jacobian_eval = jacobian_eval
            self.set_time = set_time
            # We should be solving a square system
            self.sample_residual = residual_eval(0, self.solution, self.solution_dot)
            self.sample_jacobian = jacobian_eval(0, self.solution, self.solution_dot, 0.)
            assert self.sample_jacobian.M == self.sample_jacobian.N
            assert self.sample_jacobian.N == self.sample_residual.N
            # Storage for current BC
            self.current_bc = None
            # Define an Assimulo Implicit problem
            def _store_solution_and_solution_dot(t, solution, solution_dot):
                self.solution.vector()[:] = solution.reshape((-1, 1))
                self.solution_dot.vector()[:] = solution_dot.reshape((-1, 1))
                # Update current bc
                if self.bc_eval is not None:
                    bcs_t = self.bc_eval(t)
                    assert isinstance(bcs_t, (tuple, dict))
                    if isinstance(bcs_t, tuple):
                        self.current_bc = DirichletBC(bcs_t)
                    elif isinstance(bcs_t, dict):
                        self.current_bc = DirichletBC(bcs_t, self.solution.vector()._basis_component_index_to_component_name, self.solution.vector().N)
                    else:
                        raise TypeError("Invalid bc in _LinearSolver.__init__().")
            def _assimulo_residual_eval(t, solution, solution_dot):
                # Store current time
                self.set_time(t)
                # Convert to a matrix with one column, rather than an array
                _store_solution_and_solution_dot(t, solution, solution_dot)
                # Compute residual
                residual_vector = self.residual_eval(t, self.solution, self.solution_dot)
                # Apply BCs, if necessary
                if self.bc_eval is not None:
                    self.current_bc.apply_to_vector(residual_vector, self.solution.vector())
                # Convert to an array, rather than a matrix with one column, and return
                return asarray(residual_vector).reshape(-1)
            def _assimulo_jacobian_eval(solution_dot_coefficient, t, solution, solution_dot):
                # Store current time
                self.set_time(t)
                # Convert to a matrix with one column, rather than an array
                _store_solution_and_solution_dot(t, solution, solution_dot)
                # Compute jacobian
                jacobian_matrix = self.jacobian_eval(t, self.solution, self.solution_dot, solution_dot_coefficient)
                # Apply BCs, if necessary
                if self.bc_eval is not None:
                    self.current_bc.apply_to_matrix(jacobian_matrix)
                # Return
                return jacobian_matrix
            self.problem = Implicit_Problem(_assimulo_residual_eval, self.solution.vector(), self.solution_dot.vector())
            self.problem.jac = _assimulo_jacobian_eval
            # Define an Assimulo IDA solver
            self.solver = IDA(self.problem)
            self.solver.display_progress = False
            self.solver.verbosity = 50
            # Additional storage which will be setup by set_parameters
            self._final_time = None
            self._initial_time = 0.
            self._max_time_steps = None
            self._monitor = None
            self._time_step_size = None
            
        def set_parameters(self, parameters):
            for (key, value) in parameters.items():
                if key == "absolute_tolerance":
                    self.solver.atol = value
                elif key == "final_time":
                    self._final_time = value
                elif key == "initial_time":
                    self._initial_time = value
                elif key == "integrator_type":
                    assert value == "ida"
                elif key == "max_time_steps":
                    self.solver.maxsteps = value
                    self._max_time_steps = value
                elif key == "monitor":
                    self._monitor = value
                elif key == "nonlinear_solver":
                    for (key_nonlinear, value_nonlinear) in value.items():
                        if key_nonlinear == "absolute_tolerance":
                            self.solver.atol = value_nonlinear
                        elif key_nonlinear == "line_search":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "maximum_iterations":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "relative_tolerance":
                            self.solver.rtol = value
                        elif key_nonlinear == "report":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "solution_tolerance":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        else:
                            raise ValueError("Invalid paramater passed to _AssimuloIDA object.")
                elif key == "problem_type":
                    pass
                elif key == "relative_tolerance":
                    self.solver.rtol = value
                elif key == "report":
                    self.solver.verbosity = 10
                    self.solver.display_progress = True
                    self.solver.report_continuously = True
                elif key == "time_step_size":
                    self.solver.inith = value
                    self._time_step_size = value
                else:
                    raise ValueError("Invalid paramater passed to _AssimuloIDA object.")
            
        def solve(self):
            assert self._max_time_steps is not None or self._time_step_size is not None
            if self._time_step_size is not None:
                all_t = [0]
                all_t_arange = self._time_step_size + arange(self._initial_time, self._final_time, self._time_step_size)
                all_t.extend(all_t_arange.tolist())
            elif self._max_time_steps is not None:
                all_t = linspace(self._initial_time, self._final_time, num=self._max_time_steps+1)
                all_t = all_t.tolist()
            for ida_trial in range(5):
                try:
                    all_times, all_solutions, all_solutions_dot = self.solver.simulate(self._final_time, ncp_list=all_t)
                except IDAError as error:
                    if str(error) == "'Error test failures occurred too many times during one internal time step or minimum step size was reached. At time 0.000000.'":
                        # There is no way to increase the number of error test failures in the assimulo interface, try again with smaller inith
                        self.solver.inith /= 10.
                    else:
                        # There was an error, but we cannot handle it. Raise it again
                        raise
                else:
                    break
            # Convert all_solutions to a list of Function
            all_solutions_as_functions = list()
            all_solutions_dot_as_functions = list()
            for (t, solution) in zip(all_times, all_solutions):
                self.solution.vector()[:] = solution.reshape((-1, 1))
                all_solutions_as_functions.append(function_copy(self.solution))
                if len(all_solutions_as_functions) > 1: # monitor is being called at t > 0.
                    self.solution_dot.vector()[:] = (all_solutions_as_functions[-1].vector() - all_solutions_as_functions[-2].vector())/self._time_step_size
                else:
                    self.solution_dot.vector()[:] = all_solutions_dot[0].reshape((-1, 1))
                all_solutions_dot_as_functions.append(function_copy(self.solution_dot))
                if self._monitor is not None:
                    self._monitor(t, self.solution, self.solution_dot)
            self.solution.vector()[:] = all_solutions_as_functions[-1].vector()
            self.solution_dot.vector()[:] = all_solutions_dot_as_functions[-1].vector()
            return all_times, all_solutions_as_functions, all_solutions_dot_as_functions
            
