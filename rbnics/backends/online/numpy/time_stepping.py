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

from numpy import arange, isclose, linspace
try:
    from assimulo.solvers import IDA
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
from rbnics.backends.online.numpy.nonlinear_solver import NonlinearSolver, NonlinearProblemWrapper
from rbnics.utils.decorators import BackendFor

@BackendFor("numpy", inputs=(TimeDependentProblemWrapper, Function.Type(), Function.Type(), (Function.Type(), None)))
class TimeStepping(AbstractTimeStepping):
    def __init__(self, problem_wrapper, solution, solution_dot, solution_dot_dot=None):
        assert solution_dot_dot is None
        ic = problem_wrapper.ic_eval()
        if ic is not None:
            assign(solution, ic)
        self.problem = _TimeDependentProblem(problem_wrapper.residual_eval, solution, solution_dot, problem_wrapper.bc_eval, problem_wrapper.jacobian_eval, problem_wrapper.set_time)
        self._monitor_callback = problem_wrapper.monitor
        self.solver = self.problem.create_solver({"problem_type": "linear"})
        self.solver._monitor_callback = self._monitor_callback
        
    def set_parameters(self, parameters):
        self.solver = self.problem.create_solver(parameters)
        self.solver._monitor_callback = self._monitor_callback
                
    def solve(self):
        self.solver.solve()
        
class _TimeDependentProblem(object):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_dot = solution_dot
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        
    def create_solver(self, parameters=None):
        if parameters is None:
            parameters = dict()
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
        self.solution_previous = function_copy(solution)
        self.zero = Function(solution.vector().N) # equal to zero
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        self.problem_type = problem_type
        # Setup solver
        if problem_type == "linear":
            self.minus_solution_previous_over_dt = function_copy(solution)
            class _LinearSolver(LinearSolver):
                def __init__(self_, t):
                    self.set_time(t)
                    self.minus_solution_previous_over_dt.vector()[:] = self.solution_previous.vector()
                    self.minus_solution_previous_over_dt.vector()[:] /= - self._time_step_size
                    lhs = self.jacobian_eval(t, self.zero, self.zero, 1./self._time_step_size)
                    rhs = - self.residual_eval(t, self.zero, self.minus_solution_previous_over_dt)
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
                        def monitor(self_, solution):
                            pass
                    NonlinearSolver.__init__(self_, _NonlinearProblemWrapper(), self.solution)
                
            self.solver_generator = _NonlinearSolver
        # Additional storage which will be setup by set_parameters
        self._final_time = None
        self._initial_time = 0.
        self._nonlinear_solver_parameters = None
        self._max_time_steps = None
        self._report = None
        self._monitor_callback = None
        self._monitor_initial_time = None
        self._monitor_time_step_size = None
        self._time_step_size = None
        
    def _monitor(self, t, solution, solution_dot):
        if self._monitor_callback is not None:
            self._monitor_callback(t, solution, solution_dot)
    
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
                assert isinstance(value, dict)
                assert all(key_monitor in ("initial_time", "time_step_size") for key_monitor in value)
                if "initial_time" in value:
                    self._monitor_initial_time = value["initial_time"]
                if "time_step_size" in value:
                    self._monitor_time_step_size = value["time_step_size"]
            elif key == "nonlinear_solver":
                self._nonlinear_solver_parameters = value
            elif key == "problem_type":
                assert value == self.problem_type
            elif key == "report":
                if value is True:
                    def print_time(t):
                        print("# t = {0:g}".format(t))
                    self._report = print_time
                else:
                    self._report = None
            elif key == "time_step_size":
                self._time_step_size = value
            else:
                raise ValueError("Invalid paramater passed to _ScipyImplicitEuler object.")
                
    def solve(self):
        # Prepar time array
        assert self._max_time_steps is not None or self._time_step_size is not None
        if self._time_step_size is not None:
            all_t = arange(self._initial_time, self._final_time + self._time_step_size/2., self._time_step_size)
        elif self._max_time_steps is not None:
            all_t = linspace(self._initial_time, self._final_time, num=self._max_time_steps+1)
            self._time_step_size = float(all_t[2] - all_t[1])
        else:
            raise ValueError("Time step size and maximum time steps cannot be both None")
        # Assert consistency of final time and time step size
        final_time_consistency = (all_t[-1] - all_t[0])/self._time_step_size
        assert isclose(round(final_time_consistency), final_time_consistency), "Final time should be occuring after an integer number of time steps"
        # Prepare monitor computation if not provided by parameters
        if self._monitor_initial_time is None:
            self._monitor_initial_time = all_t[0]
        monitor_initial_time_consistency = (self._monitor_initial_time - self._initial_time)/self._time_step_size
        assert isclose(round(monitor_initial_time_consistency), monitor_initial_time_consistency), "Monitor initial time should be occuring after an integer number of time steps"
        if self._monitor_time_step_size is None:
            self._monitor_time_step_size = self._time_step_size
        monitor_dt_consistency = self._monitor_time_step_size/self._time_step_size
        assert isclose(round(monitor_dt_consistency), monitor_dt_consistency), "Monitor time step size should be a multiple of the time step size"
        monitor_first_index = abs(all_t - self._monitor_initial_time).argmin()
        assert isclose(all_t[monitor_first_index], self._monitor_initial_time, atol=0.1*self._time_step_size)
        monitor_step = int(round(monitor_dt_consistency))
        monitor_t = all_t[monitor_first_index::monitor_step]
        # Solve
        if all_t[0] in monitor_t:
            self._monitor(all_t[0], self.solution, self.solution_dot)
        for t in all_t[1:]:
            if self._report is not None:
                self._report(t)
            solver = self.solver_generator(t)
            if self.problem_type == "nonlinear":
                if self._nonlinear_solver_parameters is not None:
                    solver.set_parameters(self._nonlinear_solver_parameters)
            solver.solve()
            self.solution_dot.vector()[:] = (self.solution.vector() - self.solution_previous.vector())/self._time_step_size
            if t in monitor_t:
                self._monitor(t, self.solution, self.solution_dot)
            self.solution_previous.vector()[:] = self.solution.vector()
        
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
            # Additional storage which will be setup by set_parameters
            self._absolute_tolerance = None
            self._final_time = None
            self._initial_time = 0.
            self._max_time_steps = None
            self._monitor_callback = None
            self._monitor_initial_time = None
            self._monitor_time_step_size = None
            self._relative_tolerance = None
            self._report = False
            self._time_step_size = None
            
        def _update_bcs(self, t):
            # Update current bc
            bcs_t = self.bc_eval(t)
            assert isinstance(bcs_t, (tuple, dict))
            if isinstance(bcs_t, tuple):
                self.current_bc = DirichletBC(bcs_t)
            elif isinstance(bcs_t, dict):
                self.current_bc = DirichletBC(bcs_t, self.sample_residual._component_name_to_basis_component_index, self.solution.vector().N)
            else:
                raise TypeError("Invalid bc in _LinearSolver.__init__().")
                    
        def _residual_vector_eval(self, t, solution, solution_dot):
            # Store current time
            self.set_time(t)
            # Store solution and solution_dot
            self.solution.vector()[:] = solution
            self.solution_dot.vector()[:] = solution_dot
            # Compute residual
            residual_vector = self.residual_eval(t, self.solution, self.solution_dot)
            # Apply BCs, if necessary
            if self.bc_eval is not None:
                self._update_bcs(t)
                self.current_bc.apply_to_vector(residual_vector, self.solution.vector())
            # Convert to an array, rather than a matrix with one column, and return
            return residual_vector.__array__()
            
        def _jacobian_matrix_eval(self, solution_dot_coefficient, t, solution, solution_dot):
            # Store current time
            self.set_time(t)
            # Store solution and solution_dot
            self.solution.vector()[:] = solution
            self.solution_dot.vector()[:] = solution_dot
            # Compute jacobian
            jacobian_matrix = self.jacobian_eval(t, self.solution, self.solution_dot, solution_dot_coefficient)
            # Apply BCs, if necessary
            if self.bc_eval is not None:
                self._update_bcs(t)
                self.current_bc.apply_to_matrix(jacobian_matrix)
            # Return
            return jacobian_matrix.__array__()
            
        def _monitor(self, solver, t, solution, solution_dot):
            # Store solution and solution_dot
            self.solution.vector()[:] = solution
            self.solution_dot.vector()[:] = solution_dot
            # Call monitor
            if self._monitor_callback is not None:
                self._monitor_callback(t, self.solution, self.solution_dot)
            
        def set_parameters(self, parameters):
            for (key, value) in parameters.items():
                if key == "absolute_tolerance":
                    self._absolute_tolerance = value
                elif key == "final_time":
                    self._final_time = value
                elif key == "initial_time":
                    self._initial_time = value
                elif key == "integrator_type":
                    assert value == "ida"
                elif key == "max_time_steps":
                    self._max_time_steps = value
                elif key == "monitor":
                    assert isinstance(value, dict)
                    assert all(key_monitor in ("initial_time", "time_step_size") for key_monitor in value)
                    if "initial_time" in value:
                        self._monitor_initial_time = value["initial_time"]
                    if "time_step_size" in value:
                        self._monitor_time_step_size = value["time_step_size"]
                elif key == "nonlinear_solver":
                    for (key_nonlinear, value_nonlinear) in value.items():
                        if key_nonlinear == "absolute_tolerance":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "line_search":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "maximum_iterations":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "relative_tolerance":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "report":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        elif key_nonlinear == "solution_tolerance":
                            raise NotImplementedError("This feature has not been implemented in IDA.")
                        else:
                            raise ValueError("Invalid paramater passed to _AssimuloIDA object.")
                elif key == "problem_type":
                    pass
                elif key == "relative_tolerance":
                    self._relative_tolerance = value
                elif key == "report":
                    self._report = True
                elif key == "time_step_size":
                    self._time_step_size = value
                else:
                    raise ValueError("Invalid paramater passed to _AssimuloIDA object.")
        
        def solve(self):
            # Setup IDA
            assert self._initial_time is not None
            problem = Implicit_Problem(self._residual_vector_eval, self.solution.vector(), self.solution_dot.vector(), self._initial_time)
            problem.jac = self._jacobian_matrix_eval
            problem.handle_result = self._monitor
            # Define an Assimulo IDA solver
            solver = IDA(problem)
            # Setup options
            assert self._time_step_size is not None
            solver.inith = self._time_step_size
            if self._absolute_tolerance is not None:
                solver.atol = self._absolute_tolerance
            if self._max_time_steps is not None:
                solver.maxsteps = self._max_time_steps
            if self._relative_tolerance is not None:
                solver.rtol = self._relative_tolerance
            if self._report:
                solver.verbosity = 10
                solver.display_progress = True
                solver.report_continuously = True
            else:
                solver.display_progress = False
                solver.verbosity = 50
            # Assert consistency of final time and time step size
            assert self._final_time is not None
            final_time_consistency = (self._final_time - self._initial_time)/self._time_step_size
            assert isclose(round(final_time_consistency), final_time_consistency), "Final time should be occuring after an integer number of time steps"
            # Prepare monitor computation if not provided by parameters
            if self._monitor_initial_time is None:
                self._monitor_initial_time = self._initial_time
            assert isclose(round(self._monitor_initial_time/self._time_step_size), self._monitor_initial_time/self._time_step_size), "Monitor initial time should be a multiple of the time step size"
            if self._monitor_time_step_size is None:
                self._monitor_time_step_size = self._time_step_size
            assert isclose(round(self._monitor_time_step_size/self._time_step_size), self._monitor_time_step_size/self._time_step_size), "Monitor time step size should be a multiple of the time step size"
            monitor_t = arange(self._monitor_initial_time, self._final_time + self._monitor_time_step_size/2., self._monitor_time_step_size)
            # Solve
            solver.simulate(self._final_time, ncp_list=monitor_t)
            # Solution and solution_dot at the final time are already up to date through
            # the monitor function
