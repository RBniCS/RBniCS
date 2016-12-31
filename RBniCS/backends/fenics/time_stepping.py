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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import types
from numpy import isclose
from petsc4py import PETSc
from ufl import Form
from dolfin import as_backend_type, assemble, Function as PETScFunction, GenericMatrix, GenericVector, PETScMatrix, PETScVector
from RBniCS.backends.abstract import TimeStepping as AbstractTimeStepping
from RBniCS.backends.fenics.function import Function
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractTimeStepping)
@BackendFor("fenics", inputs=(types.FunctionType, Function.Type(), types.FunctionType, (types.FunctionType, None)))
class TimeStepping(AbstractTimeStepping):
    @override
    def __init__(self, jacobian_eval, solution, residual_eval, bcs_eval=None, time_order=1, solution_dot=None):
        """
            Signatures:
                if time_order == 1:
                    def jacobian_eval(t, solution, solution_dot, solution_dot_coefficient):
                        return Constant(solution_dot_coefficient)*(u*v)*dx + inner(grad(u), grad(v))*dx
                        
                    def residual_eval(t, solution, solution_dot):
                        return solution_dot*v*dx + inner(grad(solution), grad(v))*dx  - f*v*dx
                elif time_order == 2:
                    def jacobian_eval(t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient):
                        return Constant(solution_dot_dot_coefficient)*(u*v)*dx + inner(grad(u), grad(v))*dx
                        
                    def residual_eval(t, solution, solution_dot, solution_dot_dot):
                        return solution_dot_dot*v*dx + inner(grad(solution), grad(v))*dx  - f*v*dx  
                
                def bcs_eval(t):
                    return [DirichletBC(V, Constant(0.), boundary)]
        """
        assert time_order in (1, 2)
        if time_order == 1:
            assert solution_dot is None
            self.problem = _TimeDependentProblem1(residual_eval, solution, bcs_eval, jacobian_eval)
            self.solver  = _PETScTSIntegrator(self.problem)
        elif time_order == 2:
            if solution_dot is None:
                solution_dot = Function(solution.function_space()) # equal to zero
            self.problem = _TimeDependentProblem2(residual_eval, solution, solution_dot, bcs_eval, jacobian_eval)
            self.solver  = _PETScTSIntegrator(self.problem)
        else:
            raise AssertionError("Invalid time order in TimeStepping.__init__().")
        # Store solution input
        self.solution = solution
        # Store time order input
        self.time_order = time_order
            
    @override
    def set_parameters(self, parameters):
        self.solver.set_parameters(parameters)
        
    @override
    def solve(self):
        if self.time_order == 1:
            (all_solutions_time, all_solutions, all_solutions_dot) = self.solver.solve()
        elif self.time_order == 2:
            (all_solutions_time, all_solutions, all_solutions_dot, all_solutions_dot_dot) = self.solver.solve()
        else:
            raise AssertionError("Invalid time order in TimeStepping.solve().")
        self.solution.vector().zero()
        self.solution.vector().add_local(all_solutions[-1].vector().array())
        self.solution.vector().apply("")
        if self.time_order == 1:
            return (all_solutions_time, all_solutions, all_solutions_dot)
        elif self.time_order == 2:
            return (all_solutions_time, all_solutions, all_solutions_dot, all_solutions_dot_dot)
        else:
            raise AssertionError("Invalid time order in TimeStepping.solve().")
        
class _TimeDependentProblem_Base(object):
    def __init__(self, residual_eval, solution, bc_eval, jacobian_eval):
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        # Storage for derivatives
        self.V = solution.function_space()
        # Storage for residual and jacobian
        self.residual_vector = PETScVector()
        self.jacobian_matrix = PETScMatrix()        
        # Storage for solutions
        self.all_solutions_time = list()
        self.all_solutions = list()
        self.all_solutions_dot = list()
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
                output_solution_petsc = as_backend_type(output_solution.vector()).vec()
                ts.interpolate(self.output_t, output_solution_petsc)
                output_solution_petsc.assemble()
                output_solution_petsc.ghostUpdate()
                self.all_solutions.append(output_solution)
                # Compute time derivative by a simple finite difference
                output_solution_dot = self.all_solutions[-1].copy(deepcopy=True)
                if len(self.all_solutions) == 1: # monitor is being called at t = 0.
                    output_solution_dot.vector().zero()
                else:
                    output_solution_dot.vector().add_local(- self.all_solutions[-2].vector().array())
                    output_solution_dot.vector().apply("")
                    output_solution_dot.vector()[:] *= 1./self.output_dt
                self.all_solutions_dot.append(output_solution_dot)
            else:
                # ts.interpolate is not yet available for TSALPHA2, assume that no adaptation was carried out
                (output_solution_petsc, output_solution_dot_petsc) = ts.getSolution2()
                output_solution_petsc.assemble()
                output_solution_petsc.ghostUpdate()
                output_solution = PETScFunction(self.V, PETScVector(output_solution_petsc))
                output_solution = output_solution.copy(deepcopy=True)
                self.all_solutions.append(output_solution)
                output_solution_dot_petsc.assemble()
                output_solution_dot_petsc.ghostUpdate()
                output_solution_dot = PETScFunction(self.V, PETScVector(output_solution_dot_petsc))
                output_solution_dot = output_solution_dot.copy(deepcopy=True)
                self.all_solutions_dot.append(output_solution_dot)
                # Compute time derivative by a simple finite difference
                output_solution_dot_dot = self.all_solutions_dot[-1].copy(deepcopy=True)
                if len(self.all_solutions_dot) == 1: # monitor is being called at t = 0.
                    output_solution_dot_dot.vector().zero()
                else:
                    output_solution_dot_dot.vector().add_local(- self.all_solutions_dot[-2].vector().array())
                    output_solution_dot_dot.vector().apply("")
                    output_solution_dot_dot.vector()[:] *= 1./self.output_dt
                self.all_solutions_dot_dot.append(output_solution_dot_dot)
            if self.output_monitor is not None:
                self.output_monitor(self.output_t, output_solution)
            self.output_t_prev = self.output_t
            self.output_t += self.output_dt
            # Disable final timestep workaround
            at_final_time_step = False
        
class _TimeDependentProblem1(_TimeDependentProblem_Base):
    def __init__(self, residual_eval, solution, bc_eval, jacobian_eval):
        _TimeDependentProblem_Base.__init__(self, residual_eval, solution, bc_eval, jacobian_eval)
        # Auxiliary storage for time order
        self.time_order = 1
        # Additional storage for derivatives
        self.solution_dot = Function(self.V)
        # Make sure that residual vector and jacobian matrix are properly initialized
        self.residual_vector_assemble(0., self.solution, self.solution_dot, overwrite=True)
        self.jacobian_matrix_assemble(0., self.solution, self.solution_dot, 0., overwrite=True)
   
    def residual_vector_eval(self, ts, t, solution, solution_dot, residual):
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
        # 1. Store solution and solution_dot in FEniCS data structures
        self.update_solution(solution, solution_dot)
        # 2. Assemble the residual
        self.residual_vector = PETScVector(residual)
        self.residual_vector_assemble(t, self.solution, self.solution_dot)
        # 3. Apply boundary conditions
        for bc in self.bc_eval(t):
            bc.apply(self.residual_vector, self.solution.vector())
            
    def residual_vector_assemble(self, t, solution, solution_dot, overwrite=False):
        residual_form_or_vector = self.residual_eval(t, solution, solution_dot)
        assert isinstance(residual_form_or_vector, (Form, GenericVector))
        if isinstance(residual_form_or_vector, Form):
            assemble(residual_form_or_vector, tensor=self.residual_vector)
        elif isinstance(residual_form_or_vector, GenericVector):
            if overwrite:
                self.residual_vector = residual_form_or_vector
            else:
                as_backend_type(residual_form_or_vector).vec().copy(as_backend_type(self.residual_vector).vec())
        else:
            raise AssertionError("Invalid time order in _TimeDependentProblem1.residual_vector_assemble.")
        
    def jacobian_matrix_eval(self, ts, t, solution, solution_dot, solution_dot_coefficient, jacobian, preconditioner):
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
        # 1. Store solution and solution_dot in FEniCS data structures
        self.update_solution(solution, solution_dot)
        # 2. Assemble the jacobian
        assert jacobian == preconditioner
        self.jacobian_matrix = PETScMatrix(jacobian)
        self.jacobian_matrix_assemble(t, self.solution, self.solution_dot, solution_dot_coefficient)
        # 3. Apply boundary conditions
        for bc in self.bc_eval(t):
            bc.apply(self.jacobian_matrix)
            
    def jacobian_matrix_assemble(self, t, solution, solution_dot, solution_dot_coefficient, overwrite=False):
        jacobian_form_or_matrix = self.jacobian_eval(t, solution, solution_dot, solution_dot_coefficient)
        assert isinstance(jacobian_form_or_matrix, (Form, GenericMatrix))
        if isinstance(jacobian_form_or_matrix, Form):
            assemble(jacobian_form_or_matrix, tensor=self.jacobian_matrix)
        elif isinstance(jacobian_form_or_matrix, GenericMatrix):
            if overwrite:
                self.jacobian_matrix = jacobian_form_or_matrix
            else:
                as_backend_type(jacobian_form_or_matrix).mat().copy(as_backend_type(self.jacobian_matrix).mat())
            # Make sure to keep nonzero patter, as FEniCS does by default
            as_backend_type(jacobian_form_or_matrix).mat().setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        else:
            raise AssertionError("Invalid time order in _TimeDependentProblem1.jacobian_matrix_assemble.")
            
    def update_solution(self, solution, solution_dot):
        solution.ghostUpdate()
        solution_dot.ghostUpdate()
        self.solution = PETScFunction(self.V, PETScVector(solution))
        self.solution_dot = PETScFunction(self.V, PETScVector(solution_dot))
        
class _TimeDependentProblem2(_TimeDependentProblem_Base):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval):
        _TimeDependentProblem_Base.__init__(self, residual_eval, solution, bc_eval, jacobian_eval)
        # Additional storage for derivatives
        self.solution_dot = solution_dot
        self.solution_dot_dot = Function(self.V)
        # Auxiliary storage for time order
        self.time_order = 2
        # Make sure that residual vector and jacobian matrix are properly initialized
        self.residual_vector_assemble(0., self.solution, self.solution_dot, self.solution_dot_dot, overwrite=True)
        self.jacobian_matrix_assemble(0., self.solution, self.solution_dot, self.solution_dot_dot, 0., 0., overwrite=True)
        # Storage for solutions 
        self.all_solutions_dot_dot = list()
        
    def residual_vector_eval(self, ts, t, solution, solution_dot, solution_dot_dot, residual):
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
        # 1. Store solution and solution_dot in FEniCS data structures
        self.update_solution(solution, solution_dot, solution_dot_dot)
        # 2. Assemble the residual
        self.residual_vector = PETScVector(residual)
        self.residual_vector_assemble(t, self.solution, self.solution_dot, self.solution_dot_dot)
        # 3. Apply boundary conditions
        for bc in self.bc_eval(t):
            bc.apply(self.residual_vector, self.solution.vector())
            
    def residual_vector_assemble(self, t, solution, solution_dot, solution_dot_dot, overwrite=False):
        residual_form_or_vector = self.residual_eval(t, solution, solution_dot, solution_dot_dot)
        assert isinstance(residual_form_or_vector, (Form, GenericVector))
        if isinstance(residual_form_or_vector, Form):
            assemble(residual_form_or_vector, tensor=self.residual_vector)
        elif isinstance(residual_form_or_vector, GenericVector):
            if overwrite:
                self.residual_vector = residual_form_or_vector
            else:
                as_backend_type(residual_form_or_vector).vec().copy(as_backend_type(self.residual_vector).vec())
        else:
            raise AssertionError("Invalid time order in _TimeDependentProblem1.residual_vector_assemble.")
        
    def jacobian_matrix_eval(self, ts, t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient, jacobian, preconditioner):
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
        # 1. Store solution and solution_dot in FEniCS data structures
        self.update_solution(solution, solution_dot, solution_dot_dot)
        # 2. Assemble the jacobian
        assert jacobian == preconditioner
        self.jacobian_matrix = PETScMatrix(jacobian)
        self.jacobian_matrix_assemble(t, self.solution, self.solution_dot, self.solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient)
        # 3. Apply boundary conditions
        for bc in self.bc_eval(t):
            bc.apply(self.jacobian_matrix)
            
    def jacobian_matrix_assemble(self, t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient, overwrite=False):
        jacobian_form_or_matrix = self.jacobian_eval(t, solution, solution_dot, solution_dot_dot, solution_dot_coefficient, solution_dot_dot_coefficient)
        assert isinstance(jacobian_form_or_matrix, (Form, GenericMatrix))
        if isinstance(jacobian_form_or_matrix, Form):
            assemble(jacobian_form_or_matrix, tensor=self.jacobian_matrix)
        elif isinstance(jacobian_form_or_matrix, GenericMatrix):
            if overwrite:
                self.jacobian_matrix = jacobian_form_or_matrix
            else:
                as_backend_type(jacobian_form_or_matrix).mat().copy(as_backend_type(self.jacobian_matrix).mat())
            # Make sure to keep nonzero patter, as FEniCS does by default
            as_backend_type(jacobian_form_or_matrix).mat().setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        else:
            raise AssertionError("Invalid time order in _TimeDependentProblem1.jacobian_matrix_assemble.")
            
    def update_solution(self, solution, solution_dot, solution_dot_dot):
        solution.ghostUpdate()
        solution_dot.ghostUpdate()
        solution_dot_dot.ghostUpdate()
        self.solution = PETScFunction(self.V, PETScVector(solution))
        self.solution_dot = PETScFunction(self.V, PETScVector(solution_dot))
        self.solution_dot_dot = PETScFunction(self.V, PETScVector(solution_dot_dot))
            
            
class _PETScTSIntegrator(object):
    def __init__(self, problem):
        self.problem = problem
        # Create PETSc's TS object
        self.ts = PETSc.TS().create(problem.solution.vector().mpi_comm())
        # ... and associate residual and jacobian
        assert problem.time_order in (1, 2)
        if problem.time_order == 1:
            self.ts.setIFunction(problem.residual_vector_eval, as_backend_type(problem.residual_vector).vec())
            self.ts.setIJacobian(problem.jacobian_matrix_eval, as_backend_type(problem.jacobian_matrix).mat())
        elif problem.time_order == 2:
            self.ts.setI2Function(problem.residual_vector_eval, as_backend_type(problem.residual_vector).vec())
            self.ts.setI2Jacobian(problem.jacobian_matrix_eval, as_backend_type(problem.jacobian_matrix).mat())
        else:
            raise AssertionError("Invalid time order in _PETScTSIntegrator.__init__().")
        # ... and monitor
        self.ts.setMonitor(problem.monitor)
        # Set sensible default values to parameters
        default_parameters = {
            "exact_final_time": "stepover",
            "integrator_type": "beuler",
            "problem_type": "linear",
            "linear_solver": "mumps",
            "report": True
        }
        self.set_parameters(default_parameters)
             
    def set_parameters(self, parameters):
        for (key, value) in parameters.iteritems():
            if key == "exact_final_time":
                self.ts.setExactFinalTime(getattr(self.ts.ExactFinalTime, value.upper()))
            elif key == "final_time":
                self.ts.setMaxTime(value)
                self.problem.output_T = value
            elif key == "initial_time":
                self.ts.setTime(value)
                self.problem.output_t_prev = value
                self.problem.output_t = value
            elif key == "integrator_type":
                self.ts.setType(getattr(self.ts.Type, value.upper()))
            elif key == "linear_solver":
                snes = self.ts.getSNES()
                ksp = snes.getKSP()
                ksp.setType('preonly')
                ksp.getPC().setType('lu')
                ksp.getPC().setFactorSolverPackage(value)
            elif key == "max_time_steps":
                self.ts.setMaxSteps(value)
            elif key == "monitor":
                self.problem.output_monitor = value
            elif key == "problem_type":
                self.ts.setProblemType(getattr(self.ts.ProblemType, value.upper()))
            elif key == "report":
                if value == True:
                    def print_time(ts):
                        t = ts.getTime()
                        dt = ts.getTimeStep()
                        print("# t = " + str(t + dt))
                    self.ts.setPreStep(print_time)
                else:
                    def do_nothing(ts):
                        pass
                    self.ts.setPreStep(do_nothing)
            elif key == "snes_solver":
                snes_tolerances = [1.e-10, 1.e-9, 1.e-16, 50]
                for (key_snes, value_snes) in value.iteritems():
                    snes = self.ts.getSNES()
                    if key_snes == "absolute_tolerance":
                        snes_tolerances[0] = value_snes
                    elif key_snes == "linear_solver":
                        ksp = snes.getKSP()
                        ksp.setType('preonly')
                        ksp.getPC().setType('lu')
                        ksp.getPC().setFactorSolverPackage(value_snes)
                    elif key_snes == "line_search":
                        raise ValueError("Line search is not wrapped yet by petsc4py")
                    elif key_snes == "maximum_iterations":
                        snes_tolerances[3] = value_snes
                    elif key_snes == "method":
                        snes.setType(value_snes)
                    elif key_snes == "relative_tolerance":
                        snes_tolerances[1] = value_snes
                    elif key_snes == "report":
                            def monitor(snes, it, fgnorm):
                                print("  " + str(it) + " SNES Function norm " + "{:e}".format(fgnorm))
                            snes.setMonitor(monitor)
                    elif key_snes == "solution_tolerance":
                        snes_tolerances[2] = value_snes
                    else:
                        raise ValueError("Invalid paramater passed to PETSc SNES object.")
                snes.setTolerances(*snes_tolerances)
            elif key == "time_step_size":
                self.ts.setTimeStep(value)
                self.problem.output_dt = value
            else:
                raise ValueError("Invalid paramater passed to PETSc TS object.")
        # Finally, read in additional options from the command line
        self.ts.setFromOptions()
        
    def solve(self):
        petsc_solution = as_backend_type(self.problem.solution.vector()).vec()
        if self.problem.time_order == 1:
            self.ts.solve(petsc_solution)
        elif self.problem.time_order == 2: # need to explicitly set the solution and solution_dot, as done in PETSc/src/ts/examples/tutorials/ex43.c
            petsc_solution_dot = as_backend_type(self.problem.solution_dot.vector()).vec()
            self.ts.setSolution2(petsc_solution, petsc_solution_dot)
            self.ts.solve(petsc_solution)
        else:
            raise AssertionError("Invalid time order in _PETScTSIntegrator.solve().")
        petsc_solution.ghostUpdate()
        if self.problem.time_order == 2:
            petsc_solution_dot.ghostUpdate()
        text_output = "Total time steps %d (%d rejected, %d SNES fails)" % (self.ts.getStepNumber(), self.ts.getStepRejections(), self.ts.getSNESFailures())
        if self.ts.getProblemType() == self.ts.ProblemType.NONLINEAR:
            text_output += ", with total %d nonlinear iterations" % (self.ts.getSNESIterations(), )
        print(text_output)
        # Double check that due small roundoff errors we may have missed the monitor at the last time step
        if not isclose(self.problem.output_t_prev, self.problem.output_T, atol=0.1*self.problem.output_dt):
            self.problem.monitor(self.ts, -1, self.problem.output_T, petsc_solution)
        # Return all solutions
        if self.problem.time_order == 1:
            return self.problem.all_solutions_time, self.problem.all_solutions, self.problem.all_solutions_dot
        elif self.problem.time_order == 2:
            return self.problem.all_solutions_time, self.problem.all_solutions, self.problem.all_solutions_dot, self.problem.all_solutions_dot_dot
        else:
            raise AssertionError("Invalid time order in _PETScTSIntegrator.solve().")
        
        
