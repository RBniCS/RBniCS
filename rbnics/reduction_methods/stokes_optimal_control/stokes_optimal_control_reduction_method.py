# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from rbnics.reduction_methods.base import LinearReductionMethod


def StokesOptimalControlReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    StokesOptimalControlReductionMethod_Base = LinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass)

    class StokesOptimalControlReductionMethod_Class(StokesOptimalControlReductionMethod_Base):

        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            StokesOptimalControlReductionMethod_Base.__init__(self, truth_problem, **kwargs)
            # I/O
            self.folder["state_supremizer_snapshots"] = os.path.join(self.folder_prefix, "snapshots")
            self.folder["adjoint_supremizer_snapshots"] = os.path.join(self.folder_prefix, "snapshots")

        # Postprocess a snapshot before adding it to the basis/snapshot matrix: also solve the supremizer problems
        def postprocess_snapshot(self, snapshot, snapshot_index):
            # Compute supremizers
            print("state supremizer solve for mu =", self.truth_problem.mu)
            state_supremizer = self.truth_problem.solve_state_supremizer(snapshot)
            self.truth_problem.export_supremizer(
                self.folder["state_supremizer_snapshots"], "truth_" + str(snapshot_index), state_supremizer,
                component="s")
            print("adjoint supremizer solve for mu =", self.truth_problem.mu)
            adjoint_supremizer = self.truth_problem.solve_adjoint_supremizer(snapshot)
            self.truth_problem.export_supremizer(
                self.folder["adjoint_supremizer_snapshots"], "truth_" + str(snapshot_index), adjoint_supremizer,
                component="r")
            # Call parent
            snapshot = StokesOptimalControlReductionMethod_Base.postprocess_snapshot(self, snapshot, snapshot_index)
            # Return a tuple
            return (snapshot, state_supremizer, adjoint_supremizer)

    # return value (a class) for the decorator
    return StokesOptimalControlReductionMethod_Class
