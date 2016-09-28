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
## @file elliptic_coercive_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.backends import ProperOrthogonalDecomposition
from RBniCS.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods.elliptic_coercive.elliptic_coercive_reduction_method import EllipticCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoercivePODGalerkinReduction
#
# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@Extends(EllipticCoerciveReductionMethod) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticCoerciveProblem, "PODGalerkin")
class EllipticCoercivePODGalerkinReduction(EllipticCoerciveReductionMethod):
    """This class implements a reduced order method based on a POD (Proper
    Orthogonal Decomposition) Galerkin approach. In particular, it
    implements the offline phase and the error analysis proper for the
    POD approach.
    
    This class provides the following methods:
    
    ##  Methods related to the offline stage
    - offline()
    - update_snapshot_matrix()
    - apply_POD()

    ## Error analysis
    - error_analysis()

    A typical usage of this class is reported in tutorial 2.

    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem):
        # Call the parent initialization
        EllipticCoerciveReductionMethod.__init__(self, truth_problem)
                
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Declare a POD object
        self.POD = ProperOrthogonalDecomposition()
        # I/O
        self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
        
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
        
        # Declare a new POD
        assert len(self.truth_problem.inner_product) == 1
        self.POD = ProperOrthogonalDecomposition(self.truth_problem.V, self.truth_problem.inner_product[0])
        
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
        
        for (run, mu) in enumerate(self.xi_train):
            print("############################## run =", run, "######################################")
            
            self.truth_problem.set_mu(mu)
            
            print("truth solve for mu =", self.truth_problem.mu)
            snapshot = self.truth_problem.solve()
            self.truth_problem.export_solution(snapshot, self.folder["snapshots"], "truth_" + str(run))
            snapshot = self.reduced_problem.postprocess_snapshot(snapshot)
            
            print("update snapshots matrix")
            self.update_snapshots_matrix(snapshot)

            print("")
            run += 1
            
        print("############################## perform POD ######################################")
        self.compute_basis_functions()
        
        print("")
        print("build reduced operators")
        self.reduced_problem.build_reduced_operators()
        
        print("")
        print("==============================================================")
        print("=             Offline phase ends                             =")
        print("==============================================================")
        print("")
        
        self._finalize_offline()
        return self.reduced_problem
        
    ## Compute basis functions performing POD
    def compute_basis_functions(self):
        (Z, N) = self.POD.apply(self.Nmax)
        self.reduced_problem.Z.enrich(Z)
        self.reduced_problem.N += N
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
        self.POD.print_eigenvalues(N)
        self.POD.save_eigenvalues_file(self.folder["post_processing"], "eigs")
        self.POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
        
    ## Update the snapshots matrix
    def update_snapshots_matrix(self, snapshot):
        self.POD.store_snapshot(snapshot)
        
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
        error_analysis_table.add_column("error_s", group_name="s", operations="mean")
        
        for (run, mu) in enumerate(self.xi_test):
            print("############################## run =", run, "######################################")
            
            self.reduced_problem.set_mu(mu)
                        
            for n in range(1, N + 1): # n = 1, ... N
                (error_analysis_table["error_u", n, run], error_analysis_table["error_s", n, run]) = self.reduced_problem.compute_error(n, **kwargs)
        
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
