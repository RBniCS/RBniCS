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
from dolfin import assemble, DirichletBC, has_pybind11, PETScMatrix, PETScVector
if has_pybind11():
    from dolfin.cpp.la import GenericMatrix, GenericVector
else:
    from dolfin import GenericMatrix, GenericVector
from rbnics.backends.abstract import TimeStepping as AbstractTimeStepping, TimeDependentProblemWrapper
from rbnics.backends.basic.wrapping.petsc_ts_integrator import BasicPETScTSIntegrator
from rbnics.backends.dolfin.assign import assign
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.wrapping import get_default_linear_solver, get_mpi_comm, to_petsc4py
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.utils.decorators import BackendFor, dict_of, list_of, ModuleWrapper, overload

backend = ModuleWrapper()
wrapping_for_wrapping = ModuleWrapper(get_default_linear_solver, get_mpi_comm, to_petsc4py)
PETScTSIntegrator = BasicPETScTSIntegrator(backend, wrapping_for_wrapping)

@BackendFor("dolfin", inputs=(TimeDependentProblemWrapper, Function.Type(), Function.Type(), (Function.Type(), None)))
class TimeStepping(AbstractTimeStepping):
    def __init__(self, problem_wrapper, solution, solution_dot, solution_dot_dot=None):
        assert problem_wrapper.time_order() in (1, 2)
        if problem_wrapper.time_order() == 1:
            assert solution_dot_dot is None
            ic = problem_wrapper.ic_eval()
            if ic is not None:
                assign(solution, ic)
            self.problem = _TimeDependentProblem1(problem_wrapper.residual_eval, solution, solution_dot, problem_wrapper.bc_eval, problem_wrapper.jacobian_eval, problem_wrapper.set_time)
            self.solver = PETScTSIntegrator(self.problem, self.problem.solution.vector().copy(), self.problem.solution_dot.vector().copy()) # create copies to avoid internal storage overwriting
        elif problem_wrapper.time_order() == 2:
            assert solution_dot_dot is not None
            ic_eval_output = problem_wrapper.ic_eval()
            assert isinstance(ic_eval_output, tuple) or ic_eval_output is None
            if ic_eval_output is not None:
                assert len(ic_eval_output) == 2
                assign(solution, ic_eval_output[0])
                assign(solution_dot, ic_eval_output[1])
            self.problem = _TimeDependentProblem2(problem_wrapper.residual_eval, solution, solution_dot, solution_dot_dot, problem_wrapper.bc_eval, problem_wrapper.jacobian_eval, problem_wrapper.set_time)
            self.solver = PETScTSIntegrator(self.problem, self.problem.solution.vector().copy(), self.problem.solution_dot.vector().copy(), self.problem.solution_dot_dot.vector().copy()) # create copies to avoid internal storage overwriting
        else:
            raise ValueError("Invalid time order in TimeStepping.__init__().")
        # Store solution input
        self.solution = solution
        self.solution_dot = solution_dot
        self.solution_dot_dot = solution_dot_dot
        # Store time order input
        self.time_order = problem_wrapper.time_order()
        # Set default linear solver
        self.set_parameters({
            "linear_solver": "default"
        })
            
    def set_parameters(self, parameters):
        self.solver.set_parameters(parameters)
        
    def solve(self):
        if self.time_order == 1:
            (all_solutions_time, all_solutions, all_solutions_dot) = self.solver.solve()
        elif self.time_order == 2:
            (all_solutions_time, all_solutions, all_solutions_dot, all_solutions_dot_dot) = self.solver.solve()
        else:
            raise ValueError("Invalid time order in TimeStepping.solve().")
        self.solution.vector().zero()
        self.solution.vector().add_local(all_solutions[-1].vector().get_local())
        self.solution.vector().apply("add")
        self.solution_dot.vector().zero()
        self.solution_dot.vector().add_local(all_solutions_dot[-1].vector().get_local())
        self.solution_dot.vector().apply("add")
        if self.solution_dot_dot is not None:
            self.solution_dot_dot.vector().zero()
            self.solution_dot_dot.vector().add_local(all_solutions_dot_dot[-1].vector().get_local())
            self.solution_dot_dot.vector().apply("add")
        if self.time_order == 1:
            return (all_solutions_time, all_solutions, all_solutions_dot)
        elif self.time_order == 2:
            return (all_solutions_time, all_solutions, all_solutions_dot, all_solutions_dot_dot)
        else:
            raise ValueError("Invalid time order in TimeStepping.solve().")
        
class _TimeDependentProblem_Base(object):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_dot = solution_dot
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        # Storage for derivatives
        self.V = solution.function_space()
        # Storage for solutions (will be setup by solver)
        self.all_solutions_time = list()
        self.all_solutions = None # will be of type TimeSeries
        self.all_solutions_dot = None # will be of type TimeSeries
        # Note: self.all_solutions_dot_dot will be defined for second order problems in child class
        self.output_dt = None
        self.output_t_prev = None
        self.output_t = None
        self.output_T = None
        self.output_monitor = None
        # Auxiliary storage for time order
        self.time_order = None
            
    def monitor(self, ts, step, time, solution):
        """
           TSMonitorSet - Sets an ADDITIONAL function that is to be used at every
           timestep to display the iteration's  progress.

           Logically Collective on TS

           Input Parameters:
                +  ts - the TS context obtained from TSCreate()
                .  monitor - monitoring routine
                .  mctx - [optional] user-defined context for private data for the
                             monitor routine (use NULL if no context is desired)
                -  monitordestroy - [optional] routine that frees monitor context
                          (may be NULL)

           Calling sequence of monitor:
                $    int monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx)

                +    ts - the TS context
                .    steps - iteration number (after the final time step the monitor routine may be called with a step of -1, this indicates the solution has been interpolated to this time)
                .    time - current time
                .    u - current iterate
                -    mctx - [optional] monitoring context
        """
        
        at_final_time_step = (step == -1)
        while (self.output_t <= time and self.output_t <= self.output_T) or at_final_time_step:
            self.all_solutions_time.append(self.output_t)
            if self.time_order == 1:
                output_solution = self.solution.copy(deepcopy=True)
                output_solution_petsc = to_petsc4py(output_solution.vector())
                ts.interpolate(self.output_t, output_solution_petsc)
                output_solution_petsc.assemble()
                output_solution_petsc.ghostUpdate()
                self.all_solutions.append(output_solution)
                # Compute time derivative by a simple finite difference
                output_solution_dot = self.all_solutions[-1].copy(deepcopy=True)
                if len(self.all_solutions) == 1: # monitor is being called at t = 0.
                    output_solution_dot.vector().zero()
                else:
                    output_solution_dot.vector().add_local(- self.all_solutions[-2].vector().get_local())
                    output_solution_dot.vector().apply("add")
                    output_solution_dot.vector()[:] *= 1./self.output_dt
                self.all_solutions_dot.append(output_solution_dot)
                if self.output_monitor is not None:
                    self.output_monitor(self.output_t, output_solution, output_solution_dot)
            else:
                # ts.interpolate is not yet available for TSALPHA2, assume that no adaptation was carried out
                (output_solution_petsc, output_solution_dot_petsc) = ts.getSolution2()
                output_solution_petsc.assemble()
                output_solution_petsc.ghostUpdate()
                output_solution = self.solution.copy(deepcopy=True)
                output_solution.vector().zero()
                output_solution.vector().add_local(output_solution_petsc.getArray())
                output_solution.vector().apply("add")
                self.all_solutions.append(output_solution)
                output_solution_dot_petsc.assemble()
                output_solution_dot_petsc.ghostUpdate()
                output_solution_dot = self.solution_dot.copy(deepcopy=True)
                output_solution_dot.vector().zero()
                output_solution_dot.vector().add_local(output_solution_dot_petsc.getArray())
                output_solution_dot.vector().apply("add")
                self.all_solutions_dot.append(output_solution_dot)
                # Compute time derivative by a simple finite difference
                output_solution_dot_dot = self.all_solutions_dot[-1].copy(deepcopy=True)
                if len(self.all_solutions_dot) == 1: # monitor is being called at t = 0.
                    output_solution_dot_dot.vector().zero()
                else:
                    output_solution_dot_dot.vector().add_local(- self.all_solutions_dot[-2].vector().get_local())
                    output_solution_dot_dot.vector().apply("add")
                    output_solution_dot_dot.vector()[:] *= 1./self.output_dt
                self.all_solutions_dot_dot.append(output_solution_dot_dot)
                if self.output_monitor is not None:
                    self.output_monitor(self.output_t, output_solution, output_solution_dot, output_solution_dot_dot)
            self.output_t_prev = self.output_t
            self.output_t += self.output_dt
            # Disable final timestep workaround
            at_final_time_step = False
        
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
        
class _TimeDependentProblem1(_TimeDependentProblem_Base):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
        _TimeDependentProblem_Base.__init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time)
        # Auxiliary storage for time order
        self.time_order = 1
        # Make sure that residual vector and jacobian matrix are properly initialized
        self.residual_vector = self._residual_vector_assemble(self.residual_eval(0., self.solution, self.solution_dot))
        self.jacobian_matrix = self._jacobian_matrix_assemble(self.jacobian_eval(0., self.solution, self.solution_dot, 0.))
   
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
        
    def jacobian_matrix_eval(self, ts, t, petsc_solution, petsc_solution_dot, solution_dot_coefficient, petsc_jacobian, petsc_preconditioner):
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
        self._jacobian_matrix_assemble(self.jacobian_eval(t, self.solution, self.solution_dot, solution_dot_coefficient), petsc_jacobian)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._jacobian_bcs_apply(bcs)
        
class _TimeDependentProblem2(_TimeDependentProblem_Base):
    def __init__(self, residual_eval, solution, solution_dot, solution_dot_dot, bc_eval, jacobian_eval, set_time):
        _TimeDependentProblem_Base.__init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time)
        # Additional storage for derivatives
        self.solution_dot_dot = solution_dot_dot
        # Auxiliary storage for time order
        self.time_order = 2
        # Make sure that residual vector and jacobian matrix are properly initialized
        self.residual_vector = self._residual_vector_assemble(self.residual_eval(0., self.solution, self.solution_dot, self.solution_dot_dot))
        self.jacobian_matrix = self._jacobian_matrix_assemble(self.jacobian_eval(0., self.solution, self.solution_dot, self.solution_dot_dot, 0., 0.))
        # Storage for solutions (will be setup by solver)
        self.all_solutions_dot_dot = None # will be of type TimeSeries
        
    def residual_vector_eval(self, ts, t, petsc_solution, petsc_solution_dot, petsc_solution_dot_dot, petsc_residual):
        """
           TSSetI2Function - Set the function to compute F(t,U,U_t,U_tt) where F = 0 is the DAE to be solved.

           Logically Collective on TS

           Input Parameters:
                +  ts  - the TS context obtained from TSCreate()
                .  F   - vector to hold the residual (or NULL to have it created internally)
                .  fun - the function evaluation routine
                -  ctx - user-defined context for private data for the function evaluation routine (may be NULL)

           Calling sequence of fun:
                $  fun(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,Vec F,ctx);

                +  t    - time at step/stage being solved
                .  U    - state vector
                .  U_t  - time derivative of state vector
                .  U_tt - second time derivative of state vector
                .  F    - function vector
                -  ctx  - [optional] user-defined context for matrix evaluation routine (may be NULL)
           
           (from PETSc/src/ts/interface/ts.c)
        """
        # 1. Store solution and solution_dot in dolfin data structures, as well as current time
        self.set_time(t)
        self.update_solution(petsc_solution)
        self.update_solution_dot(petsc_solution_dot)
        self.update_solution_dot_dot(petsc_solution_dot_dot)
        # 2. Assemble the residual
        self._residual_vector_assemble(self.residual_eval(t, self.solution, self.solution_dot, self.solution_dot_dot), petsc_residual)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._residual_bcs_apply(bcs)
            
    def jacobian_matrix_eval(self, ts, t, petsc_solution, petsc_solution_dot, petsc_solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient, petsc_jacobian, petsc_preconditioner):
        """
           TSSetI2Jacobian - Set the function to compute the matrix dF/dU + v*dF/dU_t  + a*dF/dU_tt
                where F(t,U,U_t,U_tt) is the function you provided with TSSetI2Function().

           Logically Collective on TS

           Input Parameters:
                +  ts  - the TS context obtained from TSCreate()
                .  J   - Jacobian matrix
                .  P   - preconditioning matrix for J (may be same as J)
                .  jac - the Jacobian evaluation routine
                -  ctx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

           Calling sequence of jac:
                $  jac(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,PetscReal v,PetscReal a,Mat J,Mat P,void *ctx);

                +  t    - time at step/stage being solved
                .  U    - state vector
                .  U_t  - time derivative of state vector
                .  U_tt - second time derivative of state vector
                .  v    - shift for U_t
                .  a    - shift for U_tt
                .  J    - Jacobian of G(U) = F(t,U,W+v*U,W'+a*U), equivalent to dF/dU + v*dF/dU_t  + a*dF/dU_tt
                .  P    - preconditioning matrix for J, may be same as J
                -  ctx  - [optional] user-defined context for matrix evaluation routine

           Notes:
           The matrices J and P are exactly the matrices that are used by SNES for the nonlinear solve.

           The matrix dF/dU + v*dF/dU_t + a*dF/dU_tt you provide turns out to be
           the Jacobian of G(U) = F(t,U,W+v*U,W'+a*U) where F(t,U,U_t,U_tt) = 0 is the DAE to be solved.
           The time integrator internally approximates U_t by W+v*U and U_tt by W'+a*U  where the positive "shift"
           parameters 'v' and 'a' and vectors W, W' depend on the integration method, step size, and past states.
           
           (from PETSc/src/ts/interface/ts.c)
        """
        # 1. There is no need to store solution, solution_dot and solution_dot_dot in dolfin data structures,
        #    nor current time, since this has already been done by the residual
        # 2. Assemble the jacobian
        assert petsc_jacobian == petsc_preconditioner
        self._jacobian_matrix_assemble(self.jacobian_eval(t, self.solution, self.solution_dot, self.solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient), petsc_jacobian)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._jacobian_bcs_apply(bcs)
            
    def update_solution_dot_dot(self, petsc_solution_dot_dot):
        petsc_solution_dot_dot.ghostUpdate()
        self.solution_dot_dot.vector().zero()
        self.solution_dot_dot.vector().add_local(petsc_solution_dot_dot.getArray())
        self.solution_dot_dot.vector().apply("add")
