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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import types
from numpy import arange, asarray, linspace
try:
    from assimulo.solvers import IDA
    from assimulo.solvers.sundials import IDAError
    from assimulo.problem import Implicit_Problem
except ImportError:
    has_IDA = False
else:
    has_IDA = True
from RBniCS.backends.abstract import TimeStepping as AbstractTimeStepping
from RBniCS.backends.numpy.function import Function
from RBniCS.backends.numpy.linear_solver import LinearSolver
from RBniCS.backends.numpy.matrix import Matrix
from RBniCS.backends.numpy.nonlinear_solver import NonlinearSolver
from RBniCS.backends.numpy.vector import Vector
from RBniCS.backends.numpy.wrapping import function_copy
from RBniCS.backends.numpy.wrapping_utils import DirichletBC
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractTimeStepping)
@BackendFor("NumPy", inputs=(types.FunctionType, Function.Type(), types.FunctionType, (types.FunctionType, None)))
class TimeStepping(AbstractTimeStepping):
    @override
    def __init__(self, jacobian_eval, solution, residual_eval, bcs_eval=None, time_order=1, solution_dot=None):
        """
            Signatures:
                if time_order == 1:
                    def jacobian_eval(t, solution, solution_dot, solution_dot_coefficient):
                        return matrix
                        
                    def residual_eval(t, solution, solution_dot):
                        return vector
                elif time_order == 2:
                    def jacobian_eval(t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient):
                        return matrix
                        
                    def residual_eval(t, solution, solution_dot, solution_dot_dot):
                        return vector
                
                def bcs_eval(t):
                    return tuple
        """
        assert time_order in (1, 2)
        if time_order == 1:
            assert solution_dot is None
            self.problem = _TimeDependentProblem1(residual_eval, solution, bcs_eval, jacobian_eval)
            self.solver  = self.problem.create_solver()
        elif time_order == 2:
            if solution_dot is None:
                solution_dot = Function(solution.N) # equal to zero
            self.problem = _TimeDependentProblem2(residual_eval, solution, solution_dot, bcs_eval, jacobian_eval)
            self.solver  = self.problem.create_solver()
        else:
            raise AssertionError("Invalid time order in TimeStepping.__init__().")
                        
    @override
    def set_parameters(self, parameters):
        self.solver = self.problem.create_solver(parameters)
                
    @override
    def solve(self):
        return self.solver.solve()
        
class _TimeDependentProblem1(object):
    def __init__(self, residual_eval, solution, bc_eval, jacobian_eval):
        self.residual_eval = residual_eval
        self.solution = solution
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        
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
            solver = _ScipyImplicitEuler(self.residual_eval, self.solution, self.bc_eval, self.jacobian_eval, problem_type)
            solver.set_parameters(parameters)
            return solver
        elif integrator_type == "ida":
            solver = _AssimuloIDA(self.residual_eval, self.solution, self.bc_eval, self.jacobian_eval)
            solver.set_parameters(parameters)
            return solver
        else:
            raise AssertionError("Invalid integrator type in _TimeDependentProblem_Base.create_solver().")
        
class _ScipyImplicitEuler(object):
    def __init__(self, residual_eval, solution, bc_eval, jacobian_eval, problem_type):
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_previous = Function(solution.vector().N) # equal to zero
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.problem_type = problem_type
        # Setup solver
        if problem_type == "linear":
            class _LinearSolver(LinearSolver):
                def __init__(self_, t):
                    zero = Function(self.solution.vector().N) # equal to zero
                    lhs = self.jacobian_eval(t, self.solution, zero, 1./self._time_step_size)
                    rhs = self.residual_eval(t, self.solution, zero)
                    bcs_t = DirichletBC(lhs, rhs, self.bc_eval(t))
                    bcs_t_previous = DirichletBC(lhs, rhs, self.bc_eval(t - self._time_step_size))
                    bcs_increment = bcs_t - bcs_t_previous
                    solution_increment = Function(self.solution.vector().N) # equal to zero
                    LinearSolver.__init__(self_, lhs, solution_increment, - rhs, bcs_increment.bcs)
                    
                def solve(self_):
                    solution_increment = LinearSolver.solve(self_)
                    self.solution.vector()[:] += solution_increment.vector()
                    return self.solution
                
            self.solver_generator = _LinearSolver
        elif problem_type == "nonlinear":
            class _NonlinearSolver(NonlinearSolver):
                def __init__(self_, t):
                    self_.solution = Function(self.solution.vector().N) # equal to zero
                    self_.solution_dot = Function(self.solution.vector().N) # equal to zero
                    def _store_solution_and_solution_dot(solution):
                        self_.solution.vector()[:] = solution
                        self_.solution_dot.vector()[:] = (solution - self.solution_previous.vector())/self._time_step_size
                    def _jacobian_eval(solution):
                        _store_solution_and_solution_dot(solution)
                        return self.jacobian_eval(t, self_.solution, self_.solution_dot, 1./self._time_step_size)
                    def _residual_eval(solution):
                        _store_solution_and_solution_dot(solution)
                        return self.residual_eval(t, self_.solution, self_.solution_dot)
                    bcs_t = self.bc_eval(t)
                    NonlinearSolver.__init__(self_, _jacobian_eval, self.solution, _residual_eval, bcs_t)
                
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
        for (key, value) in parameters.iteritems():
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
            all_t = self._time_step_size + arange(self._initial_time, self._final_time, self._time_step_size)
        elif self._max_time_steps is not None:
            all_t = linspace(self._initial_time, self._final_time, num=self._max_time_steps+1)
            all_t = all_t[1:]
            self._time_step_size = float(all_t[2] - all_t[1])
            
        all_solutions = list()
        all_solutions.append(function_copy(self.solution))
        all_solutions_dot = list()
        all_solutions_dot.append(Function(self.solution.N))
        for t in all_t:
            if self._report is not None:
                self._report(t)
            solver = self.solver_generator(t)
            if self.problem_type == "nonlinear":
                solver.set_parameters(self._nonlinear_solver_parameters)
            self.solution_previous.vector()[:] = self.solution.vector()
            solution = solver.solve()
            if self._monitor is not None:
                self._monitor(t, solution)
            all_solutions.append(function_copy(solution))
            solution_dot = Function(self.solution.vector().N)
            solution_dot.vector()[:] = (all_solutions[-1].vector() - all_solutions[-2].vector())/self._time_step_size
            all_solutions_dot.append(solution_dot)
        
        return all_t, all_solutions, all_solutions_dot
        
if has_IDA:
    class _AssimuloIDA(object):
        def __init__(self, residual_eval, solution, bc_eval, jacobian_eval):
            self.residual_eval = residual_eval
            self.solution = solution
            self.solution_dot = Function(solution.vector().N) # equal to zero
            self.bc_eval = bc_eval
            self.jacobian_eval = jacobian_eval
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
                    self.current_bc = DirichletBC(self.sample_jacobian, self.sample_residual, self.bc_eval(t))
                    self.current_bc.apply_to_vector(self.solution.vector())
                    solution[:] = asarray(self.solution.vector()).reshape(-1)
            def _assimulo_residual_eval(t, solution, solution_dot):
                # Convert to a matrix with one column, rather than an array
                _store_solution_and_solution_dot(t, solution, solution_dot)
                # Compute residual
                residual_vector = self.residual_eval(t, self.solution, self.solution_dot)
                # Apply BCs, if necessary
                if self.bc_eval is not None:
                    self.current_bc.homogeneous_apply_to_vector(residual_vector)
                # Convert to an array, rather than a matrix with one column, and return
                return asarray(residual_vector).reshape(-1)
            def _assimulo_jacobian_eval(solution_dot_coefficient, t, solution, solution_dot):
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
            for (key, value) in parameters.iteritems():
                if key == "absolute_tolerance":
                    self.solver.atol = value
                elif key == "final_time":
                    self._final_time = value
                elif key == "initial_time":
                    assert value == 0
                elif key == "integrator_type":
                    assert value == "ida"
                elif key == "max_time_steps":
                    self.solver.maxsteps = value
                    self._max_time_steps = value
                elif key == "monitor":
                    self._monitor = value
                elif key == "nonlinear_solver":
                    for (key_nonlinear, value_nonlinear) in value.iteritems():
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
                solution_as_function = Function(self.solution.vector().N)
                solution_as_function.vector()[:] = solution.reshape((-1, 1))
                # Fix bcs
                if self.bc_eval is not None:
                    self.current_bc = DirichletBC(self.sample_jacobian, self.sample_residual, self.bc_eval(t))
                    self.current_bc.apply_to_vector(solution_as_function.vector())
                all_solutions_as_functions.append(solution_as_function)
                if self._monitor is not None:
                    self._monitor(t, solution_as_function)
                solution_dot_as_function = Function(self.solution.vector().N)
                if len(all_solutions_as_functions) > 1: # monitor is being called at t > 0.
                    solution_dot_as_function.vector()[:] = (all_solutions_as_functions[-1].vector() - all_solutions_as_functions[-2].vector())/self._time_step_size
                all_solutions_dot_as_functions.append(solution_dot_as_function)
            self.solution.vector()[:] = all_solutions_as_functions[-1].vector()
            return all_times, all_solutions_as_functions, all_solutions_dot_as_functions
            
