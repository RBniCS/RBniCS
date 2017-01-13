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
## @file parabolic_coercive_rb_galerkin_reduction.py
#  @brief Implementation of a RB ROM for parabolic coercive problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import sqrt
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.base import TimeDependentRBReduction
from RBniCS.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from RBniCS.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction
from RBniCS.reduction_methods.parabolic_coercive.parabolic_coercive_reduction_method import ParabolicCoerciveReductionMethod
from RBniCS.backends import LinearSolver, ProperOrthogonalDecomposition, SnapshotsMatrix, transpose
from RBniCS.backends.online import OnlineFunction

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoerciveRBReduction
#

ParabolicCoerciveRBReduction_Base = ParabolicCoerciveReductionMethod(TimeDependentRBReduction(EllipticCoerciveRBReduction))

# Base class containing the interface of a RB ROM
# for parabolic coercive problems
@Extends(ParabolicCoerciveRBReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(ParabolicCoerciveProblem, "ReducedBasis")
class ParabolicCoerciveRBReduction(ParabolicCoerciveRBReduction_Base):
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        ParabolicCoerciveRBReduction_Base.__init__(self, truth_problem, **kwargs)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Choose among two versions of POD-Greedy
        self.POD_greedy_basis_extension = "orthogonal" # or "POD"
        #   orthogonal ~> Haasdonk, Ohlberger; ESAIM: M2AN, 2008
        #   POD ~>  Nguyen, Rozza, Patera; Calcolo, 2009
        # Declare POD objects for basis computation using POD-Greedy
        self.POD_time_trajectory = ProperOrthogonalDecomposition()
        self.POD_basis = ProperOrthogonalDecomposition()
        # POD-Greedy size
        self.N1 = 0
        self.N2 = 0
        
    ## OFFLINE: set maximum reduced space dimension (stopping criterion)
    @override
    def set_Nmax(self, Nmax, **kwargs):
        ParabolicCoerciveRBReduction_Base.set_Nmax(self, Nmax, **kwargs)
        # Set POD-Greedy sizes
        assert "POD_Greedy" in kwargs
        if self.POD_greedy_basis_extension == "POD":
            assert len(kwargs["POD_Greedy"]) == 2
            self.N1 = kwargs["POD_Greedy"][0]
            self.N2 = kwargs["POD_Greedy"][1]
        elif self.POD_greedy_basis_extension == "orthogonal":
            self.N1 = kwargs["POD_Greedy"]
        
    ## Initialize data structures required for the offline phase
    @override
    def _init_offline(self):
        # Call parent to initialize inner product
        output = ParabolicCoerciveRBReduction_Base._init_offline(self)
        
        # Check admissible values of POD_greedy_basis_extension
        assert self.POD_greedy_basis_extension in ("orthogonal", "POD")
        
        # Declare new POD object(s)
        assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
        self.POD_time_trajectory = ProperOrthogonalDecomposition(self.truth_problem.V, self.truth_problem.inner_product[0])
        if self.POD_greedy_basis_extension == "POD":
            self.POD_basis = ProperOrthogonalDecomposition(self.truth_problem.V, self.truth_problem.inner_product[0])
        
        # Return
        return output
        
    ## Update basis matrix by POD-Greedy
    def update_basis_matrix(self, snapshot):
        if self.POD_greedy_basis_extension == "POD":
            # First, compress the time trajectory stored in snapshot
            self.POD_time_trajectory.clear()
            self.POD_time_trajectory.store_snapshot(snapshot)
            (eigs1, Z1, N1) = self.POD_time_trajectory.apply(self.N1)
            self.POD_time_trajectory.print_eigenvalues(N1)
            
            # Then, compress parameter dependence (thus, we do not clear the POD object)
            self.POD_basis.store_snapshot(Z1, weight=[sqrt(e) for e in eigs1])
            (_, Z2, _) = self.POD_basis.apply(self.reduced_problem.N + self.N2)
            self.reduced_problem.Z.clear()
            self.reduced_problem.Z.enrich(Z2)
            self.reduced_problem.N += self.N2
            self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
            self.POD_basis.print_eigenvalues(self.reduced_problem.N)
            self.POD_basis.save_eigenvalues_file(self.folder["post_processing"], "eigs")
            self.POD_basis.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
            
            # Finally, we need to clear out previously computed Riesz representors, because
            # POD-Greedy basis are not hierarchical from one greedy iteration to the next one
            for qm in range(self.reduced_problem.Q["m"]):
                self.reduced_problem.riesz["m"][qm].clear()
            for qa in range(self.reduced_problem.Q["a"]):
                self.reduced_problem.riesz["a"][qa].clear()
            
        elif self.POD_greedy_basis_extension == "orthogonal":
            # Project the time trajectory on the orthogonal of the current basis
            if self.reduced_problem.N > 0:
                N = self.reduced_problem.N
                projected_solution_N = OnlineFunction(self.reduced_problem.N)
                
                assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                X = self.truth_problem.inner_product[0]
                assert len(self.reduced_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                X_N = self.reduced_problem.inner_product[0]
                
                Z = self.reduced_problem.Z
                
                orthogonal_snapshot = SnapshotsMatrix(self.truth_problem.V)
                for solution in snapshot:
                    solver = LinearSolver(X_N, projected_solution_N, transpose(Z)*X*solution)
                    solver.solve()
                    orthogonal_snapshot.enrich(solution - Z*projected_solution_N)
            else:
                orthogonal_snapshot = snapshot
                
            # Compress it using a POD
            self.POD_time_trajectory.clear()
            self.POD_time_trajectory.store_snapshot(orthogonal_snapshot)
            (eigs1, Z1, N1) = self.POD_time_trajectory.apply(self.N1)
            self.POD_time_trajectory.print_eigenvalues(N1)
            self.POD_time_trajectory.save_eigenvalues_file(self.folder["post_processing"], "eigs")
            self.POD_time_trajectory.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
            self.reduced_problem.Z.enrich(Z1)
            self.reduced_problem.N += N1
            self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
            
