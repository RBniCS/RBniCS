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

from rbnics.backends import LinearSolver, SnapshotsMatrix, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction, OnlineMatrix
from rbnics.utils.decorators import is_training_finished, PreserveClassName, ReducedProblemDecoratorFor
from .online_rectification import OnlineRectification

@ReducedProblemDecoratorFor(OnlineRectification)
def OnlineRectificationDecoratedReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):
    
    @PreserveClassName
    class OnlineRectificationDecoratedReducedProblem_Class(EllipticCoerciveReducedProblem_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Projection of truth and reduced snapshots
            self.snapshots_mu = list()
            self.snapshots = SnapshotsMatrix(truth_problem.V)
            
        def _init_operators(self, current_stage="online"):
            # Initialize additional reduced operators
            if current_stage == "online":
                self.operator["projection_truth_snapshots"] = self.assemble_operator("projection_truth_snapshots", "online")
                self.operator["projection_reduced_snapshots"] = self.assemble_operator("projection_reduced_snapshots", "online")
            elif current_stage == "offline":
                self.operator["projection_truth_snapshots"] = OnlineAffineExpansionStorage(1)
                self.operator["projection_reduced_snapshots"] = OnlineAffineExpansionStorage(1)
            # Call Parent
            EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)
            
        def build_reduced_operators(self):
            # Call Parent
            EllipticCoerciveReducedProblem_DerivedClass.build_reduced_operators(self)
            # Compute projection of truth and reduced snapshots
            self.operator["projection_truth_snapshots"] = self.assemble_operator("projection_truth_snapshots", "offline")
            self.operator["projection_reduced_snapshots"] = self.assemble_operator("projection_reduced_snapshots", "offline")
            
        def assemble_operator(self, term, current_stage="online"):
            if term == "projection_truth_snapshots" or term == "projection_reduced_snapshots":
                assert current_stage in ("online", "offline")
                if current_stage == "online": # load from file
                    self.operator[term] = OnlineAffineExpansionStorage(1)
                    self.operator[term].load(self.folder["reduced_operators"], term)
                    return self.operator[term]
                elif current_stage == "offline":
                    N = self.N
                    if term == "projection_truth_snapshots":
                        print("build projection truth snapshots for rectification")
                        assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                        X = self.truth_problem.inner_product[0]
                        assert len(self.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                        X_N = self.inner_product[0]
                        
                        Z = self.Z
                        
                        projection_truth_snapshots = OnlineMatrix({"u": N}, {"u": N})
                        for (i, snapshot_i) in enumerate(self.snapshots):
                            projected_truth_snapshot_i = OnlineFunction(N)
                            solver = LinearSolver(X_N, projected_truth_snapshot_i, transpose(Z)*X*snapshot_i)
                            solver.solve()
                            for j in range(N):
                                projection_truth_snapshots[j, i] = projected_truth_snapshot_i.vector()[j]
                        # Store and save
                        self.operator["projection_truth_snapshots"][0] = projection_truth_snapshots
                        self.operator["projection_truth_snapshots"].save(self.folder["reduced_operators"], "projection_truth_snapshots")
                    elif term == "projection_reduced_snapshots":
                        print("build projection reduced snapshots for rectification")
                        bak_mu = self.mu
                        projection_reduced_snapshots = OnlineMatrix({"u": N}, {"u": N})
                        for (i, mu_i) in enumerate(self.snapshots_mu):
                            self.set_mu(mu_i)
                            projected_reduced_snapshot_i = self.solve(N, online_rectification=False)
                            for j in range(N):
                                projection_reduced_snapshots[j, i] = projected_reduced_snapshot_i.vector()[j]
                        self.set_mu(bak_mu)
                        # Store and save
                        self.operator["projection_reduced_snapshots"][0] = projection_reduced_snapshots
                        self.operator["projection_reduced_snapshots"].save(self.folder["reduced_operators"], "projection_reduced_snapshots")
                    return self.operator[term]
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
            else:
                return EllipticCoerciveReducedProblem_DerivedClass.assemble_operator(self, term, current_stage)
            
        def _solve(self, N, **kwargs):
            if is_training_finished(self.truth_problem):
                # Temporarily change value of stabilized attribute in truth problem
                bak_stabilized = self.truth_problem.stabilized
                self.truth_problem.stabilized = kwargs.get("online_stabilization", False)
                # Solve reduced problem
                EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
                if kwargs.get("online_rectification", True):
                    intermediate_solution = OnlineFunction(N)
                    solver = LinearSolver(self.operator["projection_reduced_snapshots"][0][:N, :N], intermediate_solution, self._solution.vector())
                    solver.solve()
                    self._solution = self.operator["projection_truth_snapshots"][0][:N, :N]*intermediate_solution
                # Restore original value of stabilized attribute in truth problem
                self.truth_problem.stabilized = bak_stabilized
            else:
                EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
            
    # return value (a class) for the decorator
    return OnlineRectificationDecoratedReducedProblem_Class
