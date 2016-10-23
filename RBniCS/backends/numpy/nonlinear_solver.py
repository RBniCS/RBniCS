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
from numpy import asarray, dot
from numpy.linalg import solve
from scipy.optimize.nonlin import Jacobian, nonlin_solve
from RBniCS.backends.abstract import NonlinearSolver as AbstractNonlinearSolver
from RBniCS.backends.numpy.function import Function
from RBniCS.utils.decorators import BackendFor, DictOfThetaType, Extends, override, ThetaType

@Extends(AbstractNonlinearSolver)
@BackendFor("NumPy", inputs=(types.FunctionType, Function.Type(), types.FunctionType, ThetaType + DictOfThetaType + (None,)))
class NonlinearSolver(AbstractNonlinearSolver):
    @override
    def __init__(self, jacobian_eval, solution, residual_eval, bcs=None):
        """
            Signatures:
                def jacobian_eval(solution):
                    return matrix
                
                def residual_eval(solution):
                    return vector
        """
        self.problem = _NonlinearProblem(residual_eval, solution, bcs, jacobian_eval)
                        
    @override
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "NumPy nonlinear solver does not accept parameters yet"
                
    @override
    def solve(self):
        residual = self.problem.residual
        initial_guess_vector = asarray(self.problem.solution.vector()).reshape(-1)
        jacobian = _Jacobian(self.problem.jacobian)
        solution_vector = nonlin_solve(residual, initial_guess_vector, jacobian=jacobian, verbose=True)
        self.problem.solution.vector()[:] = solution_vector.reshape((-1, 1))
        return self.problem.solution
        
class _NonlinearProblem(object):
    def __init__(self, residual_eval, solution, bcs, jacobian_eval):
        self.residual_eval = residual_eval
        self.solution = solution
        self.bcs = bcs
        self.jacobian_eval = jacobian_eval
        # We should be solving a square system
        sample_residual = residual_eval(solution.vector())
        sample_jacobian = jacobian_eval(solution.vector())
        assert sample_jacobian.M == sample_jacobian.N
        assert sample_jacobian.N == sample_residual.N
        # Make sure to apply BCs to the initial guess, and prepare
        # additional storage for bcs if necessary
        if self.bcs is not None:
            assert isinstance(self.bcs, (tuple, dict))
            if isinstance(self.bcs, tuple):
                # No additional storage needed
                self.bcs_base_index = None
                # Apply BCs to the initial guess
                for (i, bc_i) in enumerate(self.bcs):
                    self.solution.vector()[i] = bc_i
            elif isinstance(self.bcs, dict):
                # Auxiliary dicts should have been stored in lhs and rhs, and should be consistent
                assert self.sample_residual._basis_component_index_to_component_name == self.sample_jacobian._basis_component_index_to_component_name
                assert self.sample_residual._component_name_to_basis_component_index == self.sample_jacobian._component_name_to_basis_component_index
                assert self.sample_residual._component_name_to_basis_component_length == self.sample_jacobian._component_name_to_basis_component_length
                # Fill in storage
                bcs_base_index = dict() # from component name to first index
                current_bcs_base_index = 0
                for (basis_component_index, component_name) in sorted(self.lhs._basis_component_index_to_component_name.iteritems()):
                    bcs_base_index[component_name] = current_bcs_base_index
                    current_bcs_base_index += self.rhs.N[component_name]
                self.bcs_base_index = bcs_base_index
                # Apply BCs to the initial guess
                for (component_name, component_bc) in self.bcs.iteritems():
                    for (i, bc_i) in enumerate(component_bc):
                        block_i = bcs_base_index[component_name] + i
                        self.solution.vector()[block_i] = bc_i
            else:
                raise AssertionError("Invalid bc in _NonlinearProblem.__init__().")
        
    def residual(self, solution):
        residual_vector = self.residual_eval(solution)
        # Apply BCs, if necessary
        if self.bcs is not None:
            assert isinstance(self.bcs, (tuple, dict))
            if isinstance(self.bcs, tuple):
                # Apply BCs to the increment
                for (i, _) in enumerate(self.bcs):
                    residual_vector[i] = 0.
            elif isinstance(self.bcs, dict):
                # Apply BCs to the increment
                for (component_name, component_bc) in self.bcs.iteritems():
                    for (i, _) in enumerate(component_bc):
                        block_i = bcs_base_index[component_name] + i
                        residual_vector[block_i] = 0.
            else:
                raise AssertionError("Invalid bc in _NonlinearProblem.residual().")
        # Convert to an array, rather than a matrix with one column, and return
        return asarray(residual_vector).reshape(-1)
        
    def jacobian(self, solution):
        jacobian_matrix = self.jacobian_eval(solution)
        # Apply BCs, if necessary
        if self.bcs is not None:
            assert isinstance(self.bcs, (tuple, dict))
            if isinstance(self.bcs, tuple):
                # Apply BCs
                for (i, _) in enumerate(self.bcs):
                    jacobian_matrix[i, :] = 0.
                    jacobian_matrix[i, i] = 1.
            elif isinstance(self.bcs, dict):
                # Apply BCs
                for (component_name, component_bc) in self.bcs.iteritems():
                    for (i, _) in enumerate(component_bc):
                        block_i = bcs_base_index[component_name] + i
                        jacobian_matrix[block_i, :] = 0.
                        jacobian_matrix[block_i, block_i] = 1.
            else:
                raise AssertionError("Invalid bc in _NonlinearProblem.jacobian().")
        # Return
        return jacobian_matrix
        
# Adapted from scipy/optimize/nonlin.py, asjacobian method
class _Jacobian(Jacobian):
    def __init__(self, jacobian_eval):
        self.jacobian_eval = jacobian_eval
    
    def setup(self, x, F, func):
        Jacobian.setup(self, x, F, func)
        self.x = x
    
    def update(self, x, F):
        self.x = x

    def solve(self, v, tol=0):
        J = self.jacobian_eval(self.x)
        return solve(J, v)
        
    def matvec(self, v):
        J = self.jacobian_eval(self.x)
        return dot(J, v)

    def rsolve(self, v, tol=0):
        J = self.jacobian_eval(self.x)
        return solve(J.T, v)

    def rmatvec(self, v):
        J = self.jacobian_eval(self.x)
        return dot(J.T, v)
        
