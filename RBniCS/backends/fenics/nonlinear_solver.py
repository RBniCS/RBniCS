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

from __future__ import print_function
import types
from ufl import Form
from dolfin import as_backend_type, assemble, DirichletBC, GenericMatrix, GenericVector, NonlinearProblem, PETScSNESSolver
from RBniCS.backends.abstract import NonlinearSolver as AbstractNonlinearSolver
from RBniCS.backends.fenics.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override
from RBniCS.utils.mpi import print

@Extends(AbstractNonlinearSolver)
@BackendFor("fenics", inputs=(types.FunctionType, Function.Type(), types.FunctionType, (list_of(DirichletBC), None)))
class NonlinearSolver(AbstractNonlinearSolver):
    @override
    def __init__(self, jacobian_eval, solution, residual_eval, bcs=None):
        """
            Signatures:
                def jacobian_eval(solution):
                    return grad(u)*grad(v)
                
                def residual_eval(solution):
                    return grad(solution)*grad(v)*dx - f*v*dx
        """
        problem = _NonlinearProblem(residual_eval, solution, bcs, jacobian_eval)
        self.solver  = _PETScSNESSolver(problem)
            
    @override
    def set_parameters(self, parameters):
        self.solver.parameters.update(parameters)
        
    @override
    def solve(self):
        return self.solver.solve()
    
class _NonlinearProblem(NonlinearProblem):
    def __init__(self, residual_eval, solution, bcs, jacobian_eval):
        NonlinearProblem.__init__(self)
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.bcs = bcs
        self.jacobian_eval = jacobian_eval
               
    def F(self, residual_vector, solution):
        # Assemble the residual
        self.residual_vector_assemble(residual_vector, self.solution)
        # Apply boundary conditions
        for bc in self.bcs:
            bc.apply(residual_vector, self.solution.vector())
            
    def residual_vector_assemble(self, residual_vector, solution):
        residual_form_or_vector = self.residual_eval(solution)
        assert isinstance(residual_form_or_vector, (Form, GenericVector))
        if isinstance(residual_form_or_vector, Form):
            assemble(residual_form_or_vector, tensor=residual_vector)
        elif isinstance(residual_form_or_vector, GenericVector):
            as_backend_type(residual_form_or_vector).vec().copy(as_backend_type(residual_vector).vec())
        else:
            raise AssertionError("Invalid case in _NonlinearProblem.residual_vector_assemble.")
        
    def J(self, jacobian_matrix, solution):
        # Assemble the jacobian
        self.jacobian_matrix_assemble(jacobian_matrix, self.solution)
        # Apply boundary conditions
        for bc in self.bcs:
            bc.apply(jacobian_matrix)
            
    def jacobian_matrix_assemble(self, jacobian_matrix, solution):
        jacobian_form_or_matrix = self.jacobian_eval(solution)
        assert isinstance(jacobian_form_or_matrix, (Form, GenericMatrix))
        if isinstance(jacobian_form_or_matrix, Form):
            assemble(jacobian_form_or_matrix, tensor=jacobian_matrix)
        elif isinstance(jacobian_form_or_matrix, GenericMatrix):
            as_backend_type(jacobian_form_or_matrix).mat().copy(as_backend_type(jacobian_matrix).mat())
        else:
            raise AssertionError("Invalid case in _NonlinearProblem.jacobian_matrix_assemble.")
        
class _PETScSNESSolver(PETScSNESSolver):
    def __init__(self, problem):
        PETScSNESSolver.__init__(self, problem.solution.vector().mpi_comm())
        self.problem = problem
        
    def solve(self):
        jacobian_form_or_matrix = self.problem.jacobian_eval(self.problem.solution)
        assert isinstance(jacobian_form_or_matrix, (Form, GenericMatrix))
        if isinstance(jacobian_form_or_matrix, Form):
            PETScSNESSolver.solve(self, self.problem, self.problem.solution.vector())
            return self.problem.solution
        elif isinstance(jacobian_form_or_matrix, GenericMatrix):
            ## First of all, customize Parent's init ##
            # Hack problem.J to not do anything when being called from Parent's init, 
            # because the jacobian matrix has not been initialized yet and thus cannot be copied to
            original_J = self.problem.J
            def hacked_J(problem, jacobian_matrix, solution):
                pass
            self.problem.J = types.MethodType(hacked_J, self.problem)
            # Call Parent
            PETScSNESSolver.init(self, self.problem, self.problem.solution.vector())
            # Make sure to use a matrix with proper sparsity pattern
            self.snes().setJacobian(None, as_backend_type(jacobian_form_or_matrix).mat())
            # Restore the original problem.J
            self.problem.J = original_J
            ## Then, run SNES solver. Note that we need to duplicate the code of Parent's solve  ##
            ## because Parent's init is not virtual, and thus it would not be called even if the ##
            ## previous block had been put in a overridden init method                           ##
            solution_copy = self.problem.solution.vector().copy() # due to linesearch, see Parent's code
            self.snes().solve(None, as_backend_type(solution_copy).vec())
            as_backend_type(self.problem.solution.vector()).vec().copy(as_backend_type(solution_copy).vec())
            as_backend_type(self.problem.solution.vector()).update_ghost_values()
            its = self.snes().getIterationNumber()
            reason = self.snes().getConvergedReason()
            report = self.parameters["report"]
            error_on_nonconvergence = self.parameters["error_on_nonconvergence"]
            if reason > 0 and report:
                print("PETSc SNES solver converged in " + str(its) + " iterations with convergence reason " + str(reason) + ".")
            elif reason < 0:
                print("PETSc SNES solver diverged in " + str(its) + " iterations with divergence reason " + str(reason) + ".")
            if error_on_nonconvergence and reason < 0:
                raise RuntimeError("Solver did not converge.")
            return self.problem.solution
        else:
            raise AssertionError("Invalid case in _PETScSNESSolver.solve.")
                
