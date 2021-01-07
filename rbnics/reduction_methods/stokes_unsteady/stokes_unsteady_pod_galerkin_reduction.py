# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from rbnics.backends import ProperOrthogonalDecomposition
from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.stokes_unsteady.stokes_unsteady_problem import StokesUnsteadyProblem
from rbnics.reduction_methods.base import LinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.stokes import StokesPODGalerkinReduction
from rbnics.reduction_methods.stokes_unsteady.stokes_unsteady_reduction_method import StokesUnsteadyReductionMethod


def AbstractCFDUnsteadyPODGalerkinReduction(
        AbstractCFDPODGalerkinReduction, AbstractCFDUnsteadyPODGalerkinReduction_Base):
    class AbstractCFDUnsteadyPODGalerkinReduction_Class(AbstractCFDUnsteadyPODGalerkinReduction_Base):

        # Initialize data structures required for the offline phase: overridden version because supremizer POD
        # is different from a standard component
        def _init_offline(self):
            # Call parent to initialize
            output = AbstractCFDUnsteadyPODGalerkinReduction_Base._init_offline(self)

            if self.nested_POD:
                # Declare new POD object(s)
                self.POD_time_trajectory = dict()
                for component in ("u", "p"):
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.POD_time_trajectory[component] = ProperOrthogonalDecomposition(
                        self.truth_problem.V, inner_product)
                for component in ("s", ):
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.POD_time_trajectory[component] = ProperOrthogonalDecomposition(
                        self.truth_problem.V, inner_product, component="s")

            # Return
            return output

        # Update the snapshots matrix: overridden version because supremizer POD is different from a standard component
        def update_snapshots_matrix(self, snapshot_and_supremizer_over_time):
            assert isinstance(snapshot_and_supremizer_over_time, tuple)
            assert len(snapshot_and_supremizer_over_time) == 2

            snapshot_over_time = snapshot_and_supremizer_over_time[0]
            supremizer_over_time = snapshot_and_supremizer_over_time[1]

            if self.nested_POD:
                for component in ("u", "p"):
                    (eigs1, basis_functions1) = self._nested_POD_compress_time_trajectory(
                        snapshot_over_time, component=component)
                    self.POD[component].store_snapshot(
                        basis_functions1, weight=[sqrt(e) for e in eigs1], component=component)
                for component in ("s", ):
                    (eigs1, basis_functions1) = self._nested_POD_compress_time_trajectory(
                        supremizer_over_time, component=component)
                    self.POD[component].store_snapshot(
                        basis_functions1, weight=[sqrt(e) for e in eigs1])
            else:
                # Call the steady method, which will add all snapshots and supremizers
                AbstractCFDPODGalerkinReduction.update_snapshots_matrix(
                    self, (snapshot_over_time, supremizer_over_time))

    return AbstractCFDUnsteadyPODGalerkinReduction_Class


StokesUnsteadyPODGalerkinReduction_Base = AbstractCFDUnsteadyPODGalerkinReduction(
    StokesPODGalerkinReduction,
    LinearTimeDependentPODGalerkinReduction(StokesUnsteadyReductionMethod(StokesPODGalerkinReduction)))


@ReductionMethodFor(StokesUnsteadyProblem, "PODGalerkin")
class StokesUnsteadyPODGalerkinReduction(StokesUnsteadyPODGalerkinReduction_Base):
    pass
