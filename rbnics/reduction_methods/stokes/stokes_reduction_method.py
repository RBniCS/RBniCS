# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from rbnics.reduction_methods.base import LinearReductionMethod


# Base class containing the interface of a projection based ROM
# for saddle point problems.
def StokesReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    StokesReductionMethod_Base = LinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass)

    class StokesReductionMethod_Class(StokesReductionMethod_Base):

        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            StokesReductionMethod_Base.__init__(self, truth_problem, **kwargs)
            # I/O
            self.folder["supremizer_snapshots"] = os.path.join(self.folder_prefix, "snapshots")

        # Postprocess a snapshot before adding it to the basis/snapshot matrix: also solve the supremizer problem
        def postprocess_snapshot(self, snapshot, snapshot_index):
            # Compute supremizer
            self._print_supremizer_solve_message()
            supremizer = self.truth_problem.solve_supremizer(snapshot)
            self.truth_problem.export_supremizer(self.folder["supremizer_snapshots"], "truth_" + str(snapshot_index))
            # Call parent
            snapshot = StokesReductionMethod_Base.postprocess_snapshot(self, snapshot, snapshot_index)
            # Return a tuple
            return (snapshot, supremizer)

        def _print_supremizer_solve_message(self):
            print("supremizer solve for mu =", self.truth_problem.mu)

    # return value (a class) for the decorator
    return StokesReductionMethod_Class
