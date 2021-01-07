# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearPODGalerkinReduction
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import (
    EllipticOptimalControlReductionMethod)

EllipticOptimalControlPODGalerkinReduction_Base = LinearPODGalerkinReduction(
    EllipticOptimalControlReductionMethod(DifferentialProblemReductionMethod))


@ReductionMethodFor(EllipticOptimalControlProblem, "PODGalerkin")
class EllipticOptimalControlPODGalerkinReduction(EllipticOptimalControlPODGalerkinReduction_Base):

    # Compute basis functions performing POD: overridden to handle aggregated spaces
    def compute_basis_functions(self):
        # Carry out POD
        basis_functions = dict()
        N = dict()
        for component in self.truth_problem.components:
            print("# POD for component", component)
            POD = self.POD[component]
            assert self.tol[component] == 0.
            # TODO first negelect tolerances, then compute the max of N for each aggregated pair
            (_, _, basis_functions[component], N[component]) = POD.apply(self.Nmax, self.tol[component])
            POD.print_eigenvalues(N[component])
            POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
            POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)

        # Store POD modes related to control as usual
        self.reduced_problem.basis_functions.enrich(basis_functions["u"], component="u")
        self.reduced_problem.N["u"] += N["u"]

        # Aggregate POD modes related to state and adjoint
        for component_to in ("y", "p"):
            for i in range(self.Nmax):
                # TODO should have been N[component_from] instead of Nmax, but cannot switch the next line
                for component_from in ("y", "p"):
                    self.reduced_problem.basis_functions.enrich(
                        basis_functions[component_from][i], component={component_from: component_to})
                self.reduced_problem.N[component_to] += 2

        # Save
        self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")
