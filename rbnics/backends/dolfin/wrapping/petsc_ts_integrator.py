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


from numpy import isclose
from petsc4py import PETSc
from dolfin import as_backend_type

class PETScTSIntegrator(object):
    def __init__(self, problem, solution, solution_dot, solution_dot_dot=None):
        self.problem = problem
        self.solution = solution
        self.solution_dot = solution_dot
        self.solution_dot_dot = solution_dot_dot
        # Create PETSc's TS object
        self.ts = PETSc.TS().create(as_backend_type(problem.residual_vector).vec().getComm())
        # ... and associate residual and jacobian
        assert problem.time_order in (1, 2)
        if problem.time_order == 1:
            self.ts.setIFunction(problem.residual_vector_eval, as_backend_type(problem.residual_vector).vec())
            self.ts.setIJacobian(problem.jacobian_matrix_eval, as_backend_type(problem.jacobian_matrix).mat())
        elif problem.time_order == 2:
            self.ts.setI2Function(problem.residual_vector_eval, as_backend_type(problem.residual_vector).vec())
            self.ts.setI2Jacobian(problem.jacobian_matrix_eval, as_backend_type(problem.jacobian_matrix).mat())
        else:
            raise ValueError("Invalid time order in PETScTSIntegrator.__init__().")
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
        for (key, value) in parameters.items():
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
                ksp.setType("preonly")
                ksp.getPC().setType("lu")
                ksp.getPC().setFactorSolverPackage(value)
            elif key == "max_time_steps":
                self.ts.setMaxSteps(value)
            elif key == "monitor":
                self.problem.output_monitor = value
            elif key == "problem_type":
                assert value in ("linear", "nonlinear")
                self.ts.setProblemType(getattr(self.ts.ProblemType, value.upper()))
                snes = self.ts.getSNES()
                if value == "linear":
                    snes.setType("ksponly")
                elif value == "nonlinear":
                    snes.setType("newtonls")
                else:
                    raise ValueError("Invalid paramater passed as problem type.")
            elif key == "report":
                if value is True:
                    def print_time(ts):
                        t = ts.getTime()
                        dt = ts.getTimeStep()
                        print("# t = " + str(t + dt))
                    self.ts.setPreStep(print_time)
                else:
                    def do_nothing(ts):
                        pass
                    self.ts.setPreStep(do_nothing)
                self._report = value
            elif key == "snes_solver":
                snes_tolerances = [1.e-10, 1.e-9, 1.e-16, 50]
                for (key_snes, value_snes) in value.items():
                    snes = self.ts.getSNES()
                    if key_snes == "absolute_tolerance":
                        snes_tolerances[0] = value_snes
                    elif key_snes == "linear_solver":
                        ksp = snes.getKSP()
                        ksp.setType("preonly")
                        ksp.getPC().setType("lu")
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
        petsc_solution = as_backend_type(self.solution).vec()
        if self.problem.time_order == 1:
            self.ts.solve(petsc_solution)
        elif self.problem.time_order == 2: # need to explicitly set the solution and solution_dot, as done in PETSc/src/ts/examples/tutorials/ex43.c
            petsc_solution_dot = as_backend_type(self.solution_dot).vec()
            self.ts.setSolution2(petsc_solution, petsc_solution_dot)
            self.ts.solve(petsc_solution)
        else:
            raise ValueError("Invalid time order in PETScTSIntegrator.solve().")
        petsc_solution.ghostUpdate()
        if self.problem.time_order == 2:
            petsc_solution_dot.ghostUpdate()
        text_output = "Total time steps %d (%d rejected, %d SNES fails)" % (self.ts.getStepNumber(), self.ts.getStepRejections(), self.ts.getSNESFailures())
        if self.ts.getProblemType() == self.ts.ProblemType.NONLINEAR:
            text_output += ", with total %d nonlinear iterations" % (self.ts.getSNESIterations(), )
        if self._report:
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
            raise ValueError("Invalid time order in PETScTSIntegrator.solve().")
