# Copyright (C) 2015-2019 by the RBniCS authors
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
