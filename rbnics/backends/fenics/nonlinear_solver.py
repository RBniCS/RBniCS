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

from petsc4py import PETSc
from ufl import Form
from dolfin import as_backend_type, assemble, GenericMatrix, GenericVector, NonlinearProblem, PETScSNESSolver
from rbnics.backends.abstract import NonlinearSolver as AbstractNonlinearSolver, NonlinearProblemWrapper
from rbnics.backends.fenics.function import Function
from rbnics.utils.decorators import BackendFor, Extends, override

@Extends(AbstractNonlinearSolver)
@BackendFor("fenics", inputs=(NonlinearProblemWrapper, Function.Type()))
class NonlinearSolver(AbstractNonlinearSolver):
    @override
    def __init__(self, problem_wrapper, solution):
        self.problem = _NonlinearProblem(problem_wrapper.residual_eval, solution, problem_wrapper.bc_eval(), problem_wrapper.jacobian_eval)
        self.solver  = PETScSNESSolver(solution.vector().mpi_comm())
        # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
        # Make sure to use a matrix with proper sparsity pattern if matrix_eval returns a matrix (rather than a Form)
        jacobian_form_or_matrix = self.problem.jacobian_eval(self.problem.solution)
        assert isinstance(jacobian_form_or_matrix, (Form, GenericMatrix))
        if isinstance(jacobian_form_or_matrix, GenericMatrix):
            jacobian_matrix = as_backend_type(jacobian_form_or_matrix).mat().duplicate()
            self.solver.snes().setJacobian(None, jacobian_matrix)
        # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
            
    @override
    def set_parameters(self, parameters):
        self.solver.parameters.update(parameters)
        
    @override
    def solve(self):
        self.solver.solve(self.problem, self.problem.solution.vector())
        return self.problem.solution
    
class _NonlinearProblem(NonlinearProblem):
    def __init__(self, residual_eval, solution, bcs, jacobian_eval):
        NonlinearProblem.__init__(self)
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.bcs = bcs
        self.jacobian_eval = jacobian_eval
        # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
        self._J_assemble_failed_in_init = False
        # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
               
    def F(self, residual_vector, solution):
        # Assemble the residual
        self.residual_vector_assemble(residual_vector, self.solution)
        # Apply boundary conditions
        assert isinstance(self.bcs, (dict, list))
        if isinstance(self.bcs, list):
            for bc in self.bcs:
                bc.apply(residual_vector, self.solution.vector())
        elif isinstance(self.bcs, dict):
            for key in self.bcs:
                for bc in self.bcs[key]:
                    bc.apply(residual_vector, self.solution.vector())
        else:
            raise AssertionError("Invalid type for bcs.")
            
    def residual_vector_assemble(self, residual_vector, solution):
        residual_form_or_vector = self.residual_eval(solution)
        assert isinstance(residual_form_or_vector, (Form, GenericVector))
        if isinstance(residual_form_or_vector, Form):
            assemble(residual_form_or_vector, tensor=residual_vector)
        elif isinstance(residual_form_or_vector, GenericVector):
            as_backend_type(residual_form_or_vector).vec().swap(as_backend_type(residual_vector).vec())
        else:
            raise AssertionError("Invalid case in _NonlinearProblem.residual_vector_assemble.")
        
    def J(self, jacobian_matrix, solution):
        # Assemble the jacobian
        assembled = self.jacobian_matrix_assemble(jacobian_matrix, self.solution)
        # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
        if not assembled:
            assert not self._J_assemble_failed_in_init # This should happen only once
            self._J_assemble_failed_in_init = True
            return
        # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
        # Apply boundary conditions
        assert isinstance(self.bcs, (dict, list))
        if isinstance(self.bcs, list):
            for bc in self.bcs:
                    bc.apply(jacobian_matrix)
        elif isinstance(self.bcs, dict):
            for key in self.bcs:
                for bc in self.bcs[key]:
                    bc.apply(jacobian_matrix)
        else:
            raise AssertionError("Invalid type for bcs.")
        
    def jacobian_matrix_assemble(self, jacobian_matrix, solution):
        mat = as_backend_type(jacobian_matrix).mat()
        jacobian_form_or_matrix = self.jacobian_eval(solution)
        assert isinstance(jacobian_form_or_matrix, (Form, GenericMatrix))
        if isinstance(jacobian_form_or_matrix, Form):
            assemble(jacobian_form_or_matrix, tensor=jacobian_matrix)
            return True
        elif isinstance(jacobian_form_or_matrix, GenericMatrix):
            # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
            if jacobian_matrix.empty():
                return False
            # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
            else:
                jacobian_matrix.zero()
                jacobian_matrix += jacobian_form_or_matrix
                # Make sure to keep nonzero pattern, as FEniCS does by default, because this option is apparently
                # not preserved by the sum
                as_backend_type(jacobian_matrix).mat().setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
                return True
        else:
            raise AssertionError("Invalid case in _NonlinearProblem.jacobian_matrix_assemble.")
    
