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
import random # to randomize selection in case of equal error bound
from RBniCS.linear_algebra import GramSchmidt
from RBniCS.io_utils import ErrorAnalysisTable, SpeedupAnalysisTable
from RBniCS.reduction_methods.elliptic_coercive_reduction_method import EllipticCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRB
#
# Base class containing the interface of the RB method
# for (compliant) elliptic coercive problems
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
    - get_delta()
    - get_delta_output()
    - get_eps2()
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
    def __init__(self, truth_problem):
        # Call the parent initialization
        EllipticCoerciveReductionMethod.__init__(self, truth_problem)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Declare a GS object
        self.GS = GramSchmidt(truth_problem.inner_product)
        # I/O
        self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
                
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
        
    ## Perform the offline phase of the reduced order model
    def offline(self):
        need_to_do_offline_stage = self._init_offline()
        if not need_to_do_offline_stage:
            return self.reduced_problem
                    
        print("==============================================================")
        print("=             Offline phase begins                           =")
        print("==============================================================")
        print("")
        
        for run in range(self.Nmax):
            print("############################## run = ", run, " ######################################")
            
            print("truth solve for mu = ", self.truth_problem.mu)
            snapshot = self.truth_problem.solve()
            self.truth_problem.export_solution(snapshot, self.folder["snapshots"], "truth_" + str(run))
            self.reduced_problem.postprocess_snapshot(snapshot)
            
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
            # output the delta_max
            self.greedy()

            print("")
            
        print("==============================================================")
        print("=             Offline phase ends                             =")
        print("==============================================================")
        print("")
        
        self.reduced_problem.init("online")
        return self.reduced_problem
        
    ## Update basis matrix
    def update_basis_matrix(self, snapshot):
        self.reduced_problem.Z.enrich(snapshot)
        self.GS.apply(self.reduced_problem.Z)
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis", self.truth_problem.V)
        self.reduced_problem.N += 1
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        delta_max = -1.0
        munew = None
        for mu in self.xi_train:
            self.reduced_problem.set_mu(mu)
            self.reduced_problem._solve(self.reduced_problem.N)
            delta = self.reduced_problem.get_delta()
            if (delta > delta_max or (delta == delta_max and random.random() >= 0.5)):
                delta_max = delta
                munew = mu
        assert delta_max > 0.
        assert munew is not None
        print("absolute delta max = ", delta_max)
        self.reduced_problem.set_mu(munew)
        self.save_greedy_post_processing_file(self.reduced_problem.N, delta_max, munew, self.folder["post_processing"])

    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        if N is None:
            N = self.reduced_problem.N
            
        self._init_error_analysis()
        
        print("==============================================================")
        print("=             Error analysis begins                          =")
        print("==============================================================")
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.xi_test)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("error_u", group_name="u", operations="mean")
        error_analysis_table.add_column("delta_u", group_name="u", operations="mean")
        error_analysis_table.add_column("effectivity_u", group_name="u", operations=("min", "mean", "max"))
        error_analysis_table.add_column("error_s", group_name="s", operations="mean")
        error_analysis_table.add_column("delta_s", group_name="s", operations="mean")
        error_analysis_table.add_column("effectivity_s", group_name="s", operations=("min", "mean", "max"))
        
        for run in range(len(self.xi_test)):
            print("############################## run = ", run, " ######################################")
            
            self.reduced_problem.set_mu(self.xi_test[run])
            
            for n in range(1, N + 1): # n = 1, ... N
                (current_error_u, current_error_s) = self.reduced_problem.compute_error(n)
                
                error_analysis_table["error_u", n, run] = current_error_u
                error_analysis_table["delta_u", n, run] = self.reduced_problem.get_delta()
                error_analysis_table["effectivity_u", n, run] = \
                    error_analysis_table["delta_u", n, run]/error_analysis_table["error_u", n, run]
                
                error_analysis_table["error_s", n, run] = current_error_s
                error_analysis_table["delta_s", n, run] = self.reduced_problem.get_delta_output()
                error_analysis_table["effectivity_s", n, run] = \
                    error_analysis_table["delta_s", n, run]/error_analysis_table["error_s", n, run]
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print("==============================================================")
        print("=             Error analysis ends                            =")
        print("==============================================================")
        print("")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Save greedy post processing to file
    @staticmethod
    def save_greedy_post_processing_file(N, delta_max, mu_greedy, directory):
        with open(directory + "/delta_max.txt", "a") as outfile:
            outfile.write(str(N) + " " + str(delta_max) + "\n")
        with open(directory + "/mu_greedy.txt", "a") as outfile:
            outfile.write(str(mu_greedy) + "\n")
        
    #  @}
    ########################### end - I/O - end ########################### 
        
