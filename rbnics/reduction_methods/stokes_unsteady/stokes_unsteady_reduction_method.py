# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import TimeSeries
from rbnics.reduction_methods.base import TimeDependentReductionMethod


def AbstractCFDUnsteadyReductionMethod(AbstractCFDUnsteadyReductionMethod_Base):

    class AbstractCFDUnsteadyReductionMethod_Class(AbstractCFDUnsteadyReductionMethod_Base):

        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            AbstractCFDUnsteadyReductionMethod_Base.__init__(self, truth_problem, **kwargs)
            # I/O
            self._print_supremizer_solve_message_previous_mu = None

        # Postprocess a snapshot before adding it to the basis/snapshot matrix: also solve the supremizer problem
        def postprocess_snapshot(self, snapshot_over_time, snapshot_index):
            # Call parent: this will solve for supremizers at each time step
            snapshot_and_supremizer_over_time = AbstractCFDUnsteadyReductionMethod_Base.postprocess_snapshot(
                self, snapshot_over_time, snapshot_index)
            # Convert from a time series of tuple (snapshot, supremizer) to a tuple of two lists
            # (snapshot over time, supremizer over time)
            snapshot_over_time = TimeSeries(snapshot_over_time)
            supremizer_over_time = TimeSeries(snapshot_over_time)
            for (k, snapshot_and_supremizer) in enumerate(snapshot_and_supremizer_over_time):
                assert isinstance(snapshot_and_supremizer, tuple)
                assert len(snapshot_and_supremizer) == 2
                snapshot_over_time.append(snapshot_and_supremizer[0])
                supremizer_over_time.append(snapshot_and_supremizer[1])
            # Return a tuple
            return (snapshot_over_time, supremizer_over_time)

        def _print_supremizer_solve_message(self):
            if self.truth_problem.mu != self._print_supremizer_solve_message_previous_mu:
                AbstractCFDUnsteadyReductionMethod_Base._print_supremizer_solve_message(self)
                self._print_supremizer_solve_message_previous_mu = self.truth_problem.mu

    # return value (a class) for the decorator
    return AbstractCFDUnsteadyReductionMethod_Class


# Base class containing the interface of a projection based ROM
# for saddle point problems.
def StokesUnsteadyReductionMethod(StokesReductionMethod_DerivedClass):

    StokesUnsteadyReductionMethod_Base = AbstractCFDUnsteadyReductionMethod(
        TimeDependentReductionMethod(StokesReductionMethod_DerivedClass))

    class StokesUnsteadyReductionMethod_Class(StokesUnsteadyReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return StokesUnsteadyReductionMethod_Class
