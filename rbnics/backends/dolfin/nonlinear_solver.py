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

from petsc4py import PETSc
from ufl import Form
from dolfin import assemble, DirichletBC, has_pybind11, NonlinearProblem, PETScSNESSolver
if has_pybind11():
    from dolfin.cpp.la import GenericMatrix, GenericVector
else:
    from dolfin import GenericMatrix, GenericVector
from rbnics.backends.abstract import NonlinearSolver as AbstractNonlinearSolver, NonlinearProblemWrapper
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.wrapping import to_petsc4py
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.utils.decorators import BackendFor, dict_of, list_of, overload

@BackendFor("dolfin", inputs=(NonlinearProblemWrapper, Function.Type()))
class NonlinearSolver(AbstractNonlinearSolver):
    def __init__(self, problem_wrapper, solution):
        self.problem = _NonlinearProblem(problem_wrapper.residual_eval, solution, problem_wrapper.bc_eval(), problem_wrapper.jacobian_eval)
        self.solver = PETScSNESSolver(solution.vector().mpi_comm())
        # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
        # Make sure to use a matrix with proper sparsity pattern if matrix_eval returns a matrix (rather than a Form)
        jacobian_form_or_matrix = self.problem.jacobian_eval(self.problem.solution)
        self._init_workaround(jacobian_form_or_matrix)
        # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
        
    # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
    @overload
    def _init_workaround(self, jacobian_form: Form):
        pass
        
    @overload
    def _init_workaround(self, jacobian_matrix: GenericMatrix):
        self.solver.snes().setJacobian(None, to_petsc4py(jacobian_matrix).duplicate())
    # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
            
    def set_parameters(self, parameters):
        self.solver.parameters.update(parameters)
        
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
        self.residual_bcs_apply(self.bcs, residual_vector)
            
    def residual_vector_assemble(self, residual_vector, solution):
        residual_form_or_vector = self.residual_eval(solution)
        self._residual_vector_assemble(residual_form_or_vector, residual_vector)
        
    @overload
    def _residual_vector_assemble(self, residual_form: Form, residual_vector: GenericVector):
        assemble(residual_form, tensor=residual_vector)
        
    @overload
    def _residual_vector_assemble(self, residual_vector_input: GenericVector, residual_vector_output: GenericVector):
        to_petsc4py(residual_vector_input).swap(to_petsc4py(residual_vector_output))
    
    @overload
    def residual_bcs_apply(self, bcs: None, residual_vector: GenericVector):
        pass
        
    @overload
    def residual_bcs_apply(self, bcs: (list_of(DirichletBC), ProductOutputDirichletBC), residual_vector: GenericVector):
        for bc in self.bcs:
            bc.apply(residual_vector, self.solution.vector())
        
    @overload
    def residual_bcs_apply(self, bcs: (dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC)), residual_vector: GenericVector):
        for key in self.bcs:
            for bc in self.bcs[key]:
                bc.apply(residual_vector, self.solution.vector())
        
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
        self.jacobian_bcs_apply(self.bcs, jacobian_matrix)
        
    def jacobian_matrix_assemble(self, jacobian_matrix, solution):
        jacobian_form_or_matrix = self.jacobian_eval(solution)
        return self._jacobian_matrix_assemble(jacobian_form_or_matrix, jacobian_matrix)
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_form: Form, jacobian_matrix: GenericMatrix):
        assemble(jacobian_form, tensor=jacobian_matrix)
        return True
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_matrix_input: GenericMatrix, jacobian_matrix_output: GenericMatrix):
        # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
        if jacobian_matrix_output.empty():
            return False
        # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
        else:
            jacobian_matrix_output.zero()
            jacobian_matrix_output += jacobian_matrix_input
            # Make sure to keep nonzero pattern, as dolfin does by default, because this option is apparently
            # not preserved by the sum
            to_petsc4py(jacobian_matrix_output).setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
            return True
    
    @overload
    def jacobian_bcs_apply(self, bcs: None, jacobian_matrix: GenericMatrix):
        pass
    
    @overload
    def jacobian_bcs_apply(self, bcs: (list_of(DirichletBC), ProductOutputDirichletBC), jacobian_matrix: GenericMatrix):
        for bc in self.bcs:
            bc.apply(jacobian_matrix)
    
    @overload
    def jacobian_bcs_apply(self, bcs: (dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC)), jacobian_matrix: GenericMatrix):
        for key in self.bcs:
            for bc in self.bcs[key]:
                bc.apply(jacobian_matrix)
