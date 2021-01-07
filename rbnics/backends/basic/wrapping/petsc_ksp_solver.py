# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from petsc4py import PETSc


def BasicPETScKSPSolver(backend, wrapping):

    class _BasicPETScKSPSolver(object):
        def __init__(self, lhs, solution, rhs):
            self.lhs = lhs
            self.solution = solution
            self.rhs = rhs
            self.ksp = PETSc.KSP().create(wrapping.get_mpi_comm(solution))

        def set_parameters(self, parameters):
            for (key, value) in parameters.items():
                if key == "linear_solver":
                    self.ksp.setType("preonly")
                    self.ksp.getPC().setType("lu")
                    if value == "default":
                        value = wrapping.get_default_linear_solver()
                    if hasattr(self.ksp.getPC(), "setFactorSolverType"):  # PETSc >= 3.9
                        self.ksp.getPC().setFactorSolverType(value)
                    else:
                        self.ksp.getPC().setFactorSolverPackage(value)
                else:
                    raise ValueError("Invalid paramater passed to PETSc KSP object.")
            # Finally, read in additional options from the command line
            self.ksp.setFromOptions()

        def solve(self):
            self.ksp.setOperators(wrapping.to_petsc4py(self.lhs))
            self.ksp.solve(wrapping.to_petsc4py(self.rhs), wrapping.to_petsc4py(self.solution))

    return _BasicPETScKSPSolver
