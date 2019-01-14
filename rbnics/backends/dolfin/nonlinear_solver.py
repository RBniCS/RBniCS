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

from petsc4py import PETSc
from ufl import Form
from dolfin import assemble, DirichletBC, PETScMatrix, PETScVector
from dolfin.cpp.la import GenericMatrix, GenericVector
from rbnics.backends.abstract import NonlinearSolver as AbstractNonlinearSolver, NonlinearProblemWrapper
from rbnics.backends.basic.wrapping.petsc_snes_solver import BasicPETScSNESSolver
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.wrapping import function_copy, get_default_linear_solver, get_mpi_comm, to_petsc4py
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.utils.decorators import BackendFor, dict_of, list_of, ModuleWrapper, overload

backend = ModuleWrapper()
wrapping_for_wrapping = ModuleWrapper(function_copy, get_default_linear_solver, get_mpi_comm, to_petsc4py)
PETScSNESSolver = BasicPETScSNESSolver(backend, wrapping_for_wrapping)

@BackendFor("dolfin", inputs=(NonlinearProblemWrapper, Function.Type()))
class NonlinearSolver(AbstractNonlinearSolver):
    def __init__(self, problem_wrapper, solution):
        self.problem = _NonlinearProblem(problem_wrapper.residual_eval, solution, problem_wrapper.bc_eval(), problem_wrapper.jacobian_eval)
        self.solver = PETScSNESSolver(self.problem, self.problem.solution)
        self.solver.monitor = problem_wrapper.monitor
        
    def set_parameters(self, parameters):
        self.solver.set_parameters(parameters)
        
    def solve(self):
        self.solver.solve()
    
class _NonlinearProblem(object):
    def __init__(self, residual_eval, solution, bcs, jacobian_eval):
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.bcs = bcs
        self.jacobian_eval = jacobian_eval
        # Apply boundary conditions to solution
        self._solution_bcs_apply(self.bcs)
        # Prepare storage for residual and jacobian
        self.residual_vector = self._residual_vector_assemble(residual_eval(solution))
        self.jacobian_matrix = self._jacobian_matrix_assemble(jacobian_eval(solution))
               
    def residual_vector_eval(self, snes, petsc_solution, petsc_residual):
        # 1. Store solution in dolfin data structures
        self.update_solution(petsc_solution)
        # 2. Assemble the residual
        self._residual_vector_assemble(self.residual_eval(self.solution), petsc_residual)
        # 3. Apply boundary conditions
        self._residual_bcs_apply(self.bcs)
        
    @overload
    def _residual_vector_assemble(self, residual_form: Form):
        return assemble(residual_form)
        
    @overload
    def _residual_vector_assemble(self, residual_form: Form, petsc_residual: PETSc.Vec):
        self.residual_vector = PETScVector(petsc_residual)
        assemble(residual_form, tensor=self.residual_vector)
        
    @overload
    def _residual_vector_assemble(self, residual_form: ParametrizedTensorFactory):
        return evaluate(residual_form)
        
    @overload
    def _residual_vector_assemble(self, residual_form: ParametrizedTensorFactory, petsc_residual: PETSc.Vec):
        self.residual_vector = PETScVector(petsc_residual)
        evaluate(residual_form, tensor=self.residual_vector)
        
    @overload
    def _residual_vector_assemble(self, residual_vector: GenericVector):
        return residual_vector
        
    @overload
    def _residual_vector_assemble(self, residual_vector_input: GenericVector, petsc_residual: PETSc.Vec):
        self.residual_vector = PETScVector(petsc_residual)
        to_petsc4py(residual_vector_input).swap(petsc_residual)
    
    @overload
    def _residual_bcs_apply(self, bcs: None):
        pass
        
    @overload
    def _residual_bcs_apply(self, bcs: (list_of(DirichletBC), ProductOutputDirichletBC)):
        for bc in bcs:
            bc.apply(self.residual_vector, self.solution.vector())
        
    @overload
    def _residual_bcs_apply(self, bcs: (dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC))):
        for key in bcs:
            for bc in bcs[key]:
                bc.apply(self.residual_vector, self.solution.vector())
        
    def jacobian_matrix_eval(self, snes, petsc_solution, petsc_jacobian, petsc_preconditioner):
        # 1. There is no need to store solution, since this has already been done by the residual
        # 2. Assemble the jacobian
        assert petsc_jacobian == petsc_preconditioner
        self._jacobian_matrix_assemble(self.jacobian_eval(self.solution), petsc_jacobian)
        # 3. Apply BCs, if necessary
        self._jacobian_bcs_apply(self.bcs)
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_form: Form):
        return assemble(jacobian_form)
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_form: Form, petsc_jacobian: PETSc.Mat):
        self.jacobian_matrix = PETScMatrix(petsc_jacobian)
        assemble(jacobian_form, tensor=self.jacobian_matrix)
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_form: ParametrizedTensorFactory):
        return evaluate(jacobian_form)
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_form: ParametrizedTensorFactory, petsc_jacobian: PETSc.Mat):
        self.jacobian_matrix = PETScMatrix(petsc_jacobian)
        evaluate(jacobian_form, tensor=self.jacobian_matrix)
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_matrix: GenericMatrix):
        return jacobian_matrix
        
    @overload
    def _jacobian_matrix_assemble(self, jacobian_matrix_input: GenericMatrix, petsc_jacobian: PETSc.Mat):
        self.jacobian_matrix = PETScMatrix(petsc_jacobian)
        self.jacobian_matrix.zero()
        self.jacobian_matrix += jacobian_matrix_input
        # Make sure to keep nonzero pattern, as dolfin does by default, because this option is apparently
        # not preserved by the sum
        petsc_jacobian.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
    
    @overload
    def _jacobian_bcs_apply(self, bcs: None):
        pass
    
    @overload
    def _jacobian_bcs_apply(self, bcs: (list_of(DirichletBC), ProductOutputDirichletBC)):
        for bc in bcs:
            bc.apply(self.jacobian_matrix)
    
    @overload
    def _jacobian_bcs_apply(self, bcs: (dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC))):
        for key in bcs:
            for bc in bcs[key]:
                bc.apply(self.jacobian_matrix)
                
    def update_solution(self, petsc_solution):
        petsc_solution.ghostUpdate()
        self.solution.vector().zero()
        self.solution.vector().add_local(petsc_solution.getArray())
        self.solution.vector().apply("add")
        
    @overload
    def _solution_bcs_apply(self, bcs: None):
        pass
        
    @overload
    def _solution_bcs_apply(self, bcs: (list_of(DirichletBC), ProductOutputDirichletBC)):
        for bc in bcs:
            bc.apply(self.solution.vector())
        
    @overload
    def _solution_bcs_apply(self, bcs: (dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC))):
        for key in bcs:
            for bc in bcs[key]:
                bc.apply(self.solution.vector())
