# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import GramSchmidt
from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.stokes.stokes_problem import StokesProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearRBReduction
from rbnics.reduction_methods.stokes.stokes_reduction_method import StokesReductionMethod

StokesRBReduction_Base = LinearRBReduction(StokesReductionMethod(DifferentialProblemReductionMethod))


@ReductionMethodFor(StokesProblem, "ReducedBasis")
class StokesRBReduction(StokesRBReduction_Base):

    # Initialize data structures required for the offline phase: overridden version because supremizer GS
    # is different from a standard component
    def _init_offline(self):
        # We cannot use the standard initialization provided by RBReduction because
        # supremizer GS requires a custom initialization. We thus duplicate here part of its code

        # Call parent of parent (!) to initialize inner product and reduced problem
        output = StokesRBReduction_Base._init_offline(self)

        # Declare a new GS for each basis component
        self.GS = dict()
        for component in ("u", "p"):
            inner_product = self.truth_problem.inner_product[component][0]
            self.GS[component] = GramSchmidt(self.truth_problem.V, inner_product)
        for component in ("s", ):
            inner_product = self.truth_problem.inner_product["u"][0]
            # instead of the one for "s", which has smaller size
            self.GS[component] = GramSchmidt(self.truth_problem.V, inner_product)

        # Return
        return output

    # Update the basis matrix: overridden version because the input argument now contains both snapshot and supremizer
    def update_basis_matrix(self, snapshot_and_supremizer):
        assert isinstance(snapshot_and_supremizer, tuple)
        assert len(snapshot_and_supremizer) == 2
        snapshot = snapshot_and_supremizer[0]
        supremizer = snapshot_and_supremizer[1]
        for component in ("u", "s", "p"):
            new_basis_function = self.GS[component].apply(
                supremizer if component == "s" else snapshot,
                self.reduced_problem.basis_functions[
                    component][self.reduced_problem.N_bc[component]:],
                component=component)
            self.reduced_problem.basis_functions.enrich(new_basis_function, component=component)
            self.reduced_problem.N[component] += 1
        self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")

    # Compute the error of the reduced order approximation with respect to the full order one
    # over the testing set.
    # Note that we cannot move this method to the parent class because error analysis is defined
    # by the RBReduction decorator
    def error_analysis(self, N_generator=None, filename=None, **kwargs):
        components = ["u", "p"]  # but not "s"
        kwargs["components"] = components

        StokesRBReduction_Base.error_analysis(self, N_generator, filename, **kwargs)
