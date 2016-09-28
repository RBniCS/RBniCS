# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file elliptic_coercive_rb.py
#  @brief Implementation of the reduced basis method for (compliant) elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os # for path and makedir
import shutil # for rm
from RBniCS.backends import GramSchmidt
from RBniCS.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable, GreedySelectedParametersList, GreedyErrorEstimatorsList
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods.elliptic_coercive.elliptic_coercive_reduction_method import EllipticCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRB
#
# Base class containing the interface of the RB method
# for (compliant) elliptic coercive problems
@Extends(EllipticCoerciveReductionMethod) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticCoerciveProblem, "ReducedBasis")
class EllipticCoerciveRBReduction(EllipticCoerciveReductionMethod):
    """This class implements the Certified Reduced Basis Method for
    elliptic and coercive problems. The output of interest are assumed to
    be compliant.

    During the offline stage, the parameters are chosen relying on a
    greedy algorithm. The user must specify how the alpha_lb (i.e., alpha
    lower bound) is computed since this term is needed in the a posteriori
    error estimation. RBniCS features an implementation of the Successive
    Constraints Method (SCM) for the estimation of the alpha_lb (take a
    look at tutorial 4 for the usage of SCM).
    
    The following functions are implemented:

    ## Methods related to the offline stage
    - offline()
    - update_basis_matrix()
    - greedy()
    - compute_dual_terms()
    - compute_a_dual()
    - compute_f_dual()

    ## Methods related to the online stage
    - online_output()
    - estimate_error()
    - estimate_error_output()
    - truth_output()

    ## Error analysis
    - compute_error()
    - error_analysis()
    
    ## Input/output methods
    - load_reduced_matrices()
    
    ## Problem specific methods
    - get_alpha_lb() # to be overridden

    A typical usage of this class is given in the tutorial 1.

    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem):
        # Call the parent initialization
        EllipticCoerciveReductionMethod.__init__(self, truth_problem)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Declare a GS object
        self.GS = GramSchmidt()
        # I/O
        self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
        self.greedy_selected_parameters = GreedySelectedParametersList()
        self.greedy_error_estimators = GreedyErrorEstimatorsList()
                
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    @override
    def _init_offline(self):
        # Call parent to initialize inner product
        output = EllipticCoerciveReductionMethod._init_offline(self)
        
        # Declare a new GS object
        assert len(self.truth_problem.inner_product) == 1
        self.GS = GramSchmidt(self.truth_problem.inner_product[0])
        
        # Return
        return output
        
    ## Perform the offline phase of the reduced order model
    @override
    def offline(self):
        need_to_do_offline_stage = self._init_offline()
        if not need_to_do_offline_stage:
            return self.reduced_problem
                    
        print("==============================================================")
        print("=             Offline phase begins                           =")
        print("==============================================================")
        print("")
        
        for run in range(self.Nmax):
            print("############################## run =", run, "######################################")
            
            print("truth solve for mu =", self.truth_problem.mu)
            snapshot = self.truth_problem.solve()
            self.truth_problem.export_solution(snapshot, self.folder["snapshots"], "truth_" + str(run))
            snapshot = self.reduced_problem.postprocess_snapshot(snapshot)
            
            print("update basis matrix")
            self.update_basis_matrix(snapshot)
            
            print("build reduced operators")
            self.reduced_problem.build_reduced_operators()
            
            print("reduced order solve")
            self.reduced_problem._solve(self.reduced_problem.N)
            
            print("build operators for error estimation")
            self.reduced_problem.build_error_estimation_operators()
            
            if self.reduced_problem.N < self.Nmax:
                print("find next mu")
            
            # we do a greedy even if N == Nmax in order to have in
            # output the maximum error estimator
            self.greedy()

            print("")
            
        print("==============================================================")
        print("=             Offline phase ends                             =")
        print("==============================================================")
        print("")
        
        self._finalize_offline()
        return self.reduced_problem
        
    ## Update basis matrix
    def update_basis_matrix(self, snapshot):
        self.reduced_problem.Z.enrich(snapshot)
        self.GS.apply(self.reduced_problem.Z, self.reduced_problem.N_bc)
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
        self.reduced_problem.N += 1
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        def solve_and_estimate_error(mu, index):
            self.reduced_problem.set_mu(mu)
            self.reduced_problem._solve(self.reduced_problem.N)
            return self.reduced_problem.estimate_error()
            
        (error_estimator_max, error_estimator_argmax) = self.xi_train.max(solve_and_estimate_error)
        print("maximum error estimator =", error_estimator_max)
        self.reduced_problem.set_mu(self.xi_train[error_estimator_argmax])
        self.greedy_selected_parameters.append(self.xi_train[error_estimator_argmax])
        self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
        self.greedy_error_estimators.append(error_estimator_max)
        self.greedy_error_estimators.save(self.folder["post_processing"], "error_estimator_max")

    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    @override
    def error_analysis(self, N=None, **kwargs):
        if N is None:
            N = self.reduced_problem.N
            
        self._init_error_analysis(**kwargs)
        
        print("==============================================================")
        print("=             Error analysis begins                          =")
        print("==============================================================")
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.xi_test)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("error_u", group_name="u", operations="mean")
        error_analysis_table.add_column("error_estimator_u", group_name="u", operations="mean")
        error_analysis_table.add_column("effectivity_u", group_name="u", operations=("min", "mean", "max"))
        error_analysis_table.add_column("error_s", group_name="s", operations="mean")
        error_analysis_table.add_column("error_estimator_s", group_name="s", operations="mean")
        error_analysis_table.add_column("effectivity_s", group_name="s", operations=("min", "mean", "max"))
        
        for (run, mu) in enumerate(self.xi_test):
            print("############################## run =", run, "######################################")
            
            self.reduced_problem.set_mu(mu)
            
            for n in range(1, N + 1): # n = 1, ... N
                (current_error_u, current_error_s) = self.reduced_problem.compute_error(n, **kwargs)
                
                error_analysis_table["error_u", n, run] = current_error_u
                error_analysis_table["error_estimator_u", n, run] = self.reduced_problem.estimate_error()
                error_analysis_table["effectivity_u", n, run] = \
                    error_analysis_table["error_estimator_u", n, run]/error_analysis_table["error_u", n, run]
                
                error_analysis_table["error_s", n, run] = current_error_s
                error_analysis_table["error_estimator_s", n, run] = self.reduced_problem.estimate_error_output()
                error_analysis_table["effectivity_s", n, run] = \
                    error_analysis_table["error_estimator_s", n, run]/error_analysis_table["error_s", n, run]
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print("==============================================================")
        print("=             Error analysis ends                            =")
        print("==============================================================")
        print("")
        
        self._finalize_error_analysis(**kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
        
