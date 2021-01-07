# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from petsc4py import PETSc


def BasicPETScSNESSolver(backend, wrapping):

    class _BasicPETScSNESSolver(object):
        def __init__(self, problem, solution):
            self.problem = problem
            self.solution = solution
            self.monitor = None
            # Create SNES object
            self.snes = PETSc.SNES().create(wrapping.get_mpi_comm(solution))
            # ... and associate residual and jacobian
            self.snes.setFunction(problem.residual_vector_eval, wrapping.to_petsc4py(problem.residual_vector))
            self.snes.setJacobian(problem.jacobian_matrix_eval, wrapping.to_petsc4py(problem.jacobian_matrix))
            # Set sensible default values to parameters
            self._report = None
            self.set_parameters({
                "report": True
            })

        def set_parameters(self, parameters):
            snes_tolerances = [1.e-10, 1.e-9, 1.e-16, 50]
            for (key, value) in parameters.items():
                if key == "absolute_tolerance":
                    snes_tolerances[0] = value
                elif key == "linear_solver":
                    ksp = self.snes.getKSP()
                    ksp.setType("preonly")
                    ksp.getPC().setType("lu")
                    if value == "default":
                        value = wrapping.get_default_linear_solver()
                    if hasattr(ksp.getPC(), "setFactorSolverType"):  # PETSc >= 3.9
                        ksp.getPC().setFactorSolverType(value)
                    else:
                        ksp.getPC().setFactorSolverPackage(value)
                elif key == "line_search":
                    raise ValueError("Line search is not wrapped yet by petsc4py")
                elif key == "maximum_iterations":
                    snes_tolerances[3] = value
                elif key == "method":
                    self.snes.setType(value)
                elif key == "relative_tolerance":
                    snes_tolerances[1] = value
                elif key == "report":
                    self._report = value
                    self.snes.cancelMonitor()

                    def monitor(snes, it, fgnorm):
                        print("  " + str(it) + " SNES Function norm " + "{:e}".format(fgnorm))

                    self.snes.setMonitor(monitor)
                elif key == "solution_tolerance":
                    snes_tolerances[2] = value
                else:
                    raise ValueError("Invalid paramater passed to PETSc SNES object.")
            self.snes.setTolerances(*snes_tolerances)
            # Finally, read in additional options from the command line
            self.snes.setFromOptions()

        def solve(self):
            # create copy to avoid possible internal storage overwriting by linesearch
            solution_copy = wrapping.function_copy(self.solution)
            petsc_solution_copy = wrapping.to_petsc4py(solution_copy)
            self.snes.solve(None, petsc_solution_copy)
            if self._report:
                reason = self.snes.getConvergedReason()
                its = self.snes.getIterationNumber()
                if reason > 0:
                    print("PETSc SNES solver converged in " + str(its) + " iterations with convergence reason "
                          + str(reason) + ".")
                else:
                    print("PETSc SNES solver diverged in " + str(its) + " iterations with divergence reason "
                          + str(reason) + ".")
            self.problem.update_solution(petsc_solution_copy)
            if self.monitor is not None:
                self.monitor(self.solution)

    return _BasicPETScSNESSolver
