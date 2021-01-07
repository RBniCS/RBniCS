# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from petsc4py import PETSc
from ufl import Form
from dolfin import assemble, DirichletBC, PETScMatrix, PETScVector
from dolfin.cpp.la import GenericMatrix, GenericVector
from rbnics.backends.abstract import TimeStepping as AbstractTimeStepping, TimeDependentProblemWrapper
from rbnics.backends.basic.wrapping.petsc_ts_integrator import BasicPETScTSIntegrator
from rbnics.backends.dolfin.assign import assign
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.wrapping import function_copy, get_default_linear_solver, get_mpi_comm, to_petsc4py
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.utils.decorators import BackendFor, dict_of, list_of, ModuleWrapper, overload

backend = ModuleWrapper()
wrapping_for_wrapping = ModuleWrapper(function_copy, get_default_linear_solver, get_mpi_comm, to_petsc4py)
PETScTSIntegrator = BasicPETScTSIntegrator(backend, wrapping_for_wrapping)


@BackendFor("dolfin", inputs=(TimeDependentProblemWrapper, Function.Type(), Function.Type()))
class TimeStepping(AbstractTimeStepping):
    def __init__(self, problem_wrapper, solution, solution_dot):
        ic = problem_wrapper.ic_eval()
        if ic is not None:
            assign(solution, ic)
        self.problem = _TimeDependentProblem(problem_wrapper.residual_eval, solution, solution_dot,
                                             problem_wrapper.bc_eval, problem_wrapper.jacobian_eval,
                                             problem_wrapper.set_time)
        self.solver = PETScTSIntegrator(self.problem, solution, solution_dot)
        self.solver.monitor.monitor_callback = problem_wrapper.monitor
        # Set default linear solver
        self.set_parameters({
            "linear_solver": "default"
        })

    def set_parameters(self, parameters):
        self.solver.set_parameters(parameters)

    def solve(self):
        self.solver.solve()


class _TimeDependentProblem(object):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_dot = solution_dot
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        # Make sure that residual vector and jacobian matrix are properly initialized
        self.residual_vector = self._residual_vector_assemble(self.residual_eval(
            0., self.solution, self.solution_dot))
        self.jacobian_matrix = self._jacobian_matrix_assemble(self.jacobian_eval(
            0., self.solution, self.solution_dot, 0.))

    def residual_vector_eval(self, ts, t, petsc_solution, petsc_solution_dot, petsc_residual):
        """
           TSSetIFunction - Set the function to compute F(t,U,U_t) where F() = 0 is the DAE to be solved.

           Logically Collective on TS

           Input Parameters:
                +  ts  - the TS context obtained from TSCreate()
                .  r   - vector to hold the residual (or NULL to have it created internally)
                .  f   - the function evaluation routine
                -  ctx - user-defined context for private data for the function evaluation routine (may be NULL)

           Calling sequence of f:
                $  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec F,ctx);

                +  t   - time at step/stage being solved
                .  u   - state vector
                .  u_t - time derivative of state vector
                .  F   - function vector
                -  ctx - [optional] user-defined context for matrix evaluation routine

           (from PETSc/src/ts/interface/ts.c)
        """
        # 1. Store solution and solution_dot in dolfin data structures, as well as current time
        self.set_time(t)
        self.update_solution(petsc_solution)
        self.update_solution_dot(petsc_solution_dot)
        # 2. Assemble the residual
        self._residual_vector_assemble(self.residual_eval(t, self.solution, self.solution_dot), petsc_residual)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._residual_bcs_apply(bcs)

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

    def jacobian_matrix_eval(self, ts, t, petsc_solution, petsc_solution_dot, solution_dot_coefficient,
                             petsc_jacobian, petsc_preconditioner):
        """
           TSSetIJacobian - Set the function to compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is the function
                            provided with TSSetIFunction().

           Logically Collective on TS

           Input Parameters:
                +  ts  - the TS context obtained from TSCreate()
                .  Amat - (approximate) Jacobian matrix
                .  Pmat - matrix used to compute preconditioner (usually the same as Amat)
                .  f   - the Jacobian evaluation routine
                -  ctx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

           Calling sequence of f:
                $  f(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal a,Mat Amat,Mat Pmat,void *ctx);

                +  t    - time at step/stage being solved
                .  U    - state vector
                .  U_t  - time derivative of state vector
                .  a    - shift
                .  Amat - (approximate) Jacobian of F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
                .  Pmat - matrix used for constructing preconditioner, usually the same as Amat
                -  ctx  - [optional] user-defined context for matrix evaluation routine

           Notes:
           The matrices Amat and Pmat are exactly the matrices that are used by SNES for the nonlinear solve.

           If you know the operator Amat has a null space you can use MatSetNullSpace() and MatSetTransposeNullSpace()
           to supply the null space to Amat and the KSP solvers will automatically use that null space
           as needed during the solution process.

           The matrix dF/dU + a*dF/dU_t you provide turns out to be
           the Jacobian of F(t,U,W+a*U) where F(t,U,U_t) = 0 is the DAE to be solved.
           The time integrator internally approximates U_t by W+a*U where the positive "shift"
           a and vector W depend on the integration method, step size, and past states. For example with
           the backward Euler method a = 1/dt and W = -a*U(previous timestep) so
           W + a*U = a*(U - U(previous timestep)) = (U - U(previous timestep))/dt

           (from PETSc/src/ts/interface/ts.c)
        """
        # 1. There is no need to store solution and solution_dot in dolfin data structures, nor current time,
        #    since this has already been done by the residual
        # 2. Assemble the jacobian
        assert petsc_jacobian == petsc_preconditioner
        self._jacobian_matrix_assemble(
            self.jacobian_eval(t, self.solution, self.solution_dot, solution_dot_coefficient), petsc_jacobian)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._jacobian_bcs_apply(bcs)

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

    def update_solution_dot(self, petsc_solution_dot):
        petsc_solution_dot.ghostUpdate()
        self.solution_dot.vector().zero()
        self.solution_dot.vector().add_local(petsc_solution_dot.getArray())
        self.solution_dot.vector().apply("add")
