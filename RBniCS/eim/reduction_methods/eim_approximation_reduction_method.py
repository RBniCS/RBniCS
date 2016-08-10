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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.reduction_methods.base import ReductionMethod
from RBniCS.linear_algebra import SnapshotsMatrix, OnlineMatrix
from RBniCS.utils.io import Folders, ErrorAnalysisTable, SpeedupAnalysisTable, GreedySelectedParametersList, GreedyErrorEstimatorsList
from RBniCS.utils.mpi import print, mpi_comm
from RBniCS.utils.decorators import Extends, override

#~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EIM
#
# Empirical interpolation method for the interpolation of parametrized functions
@Extends(ReductionMethod)
class EIMApproximationReductionMethod(ReductionMethod):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the EIM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, EIM_approximation, folder_prefix):
        # Call the parent initialization
        ReductionMethod.__init__(self, folder_prefix, EIM_approximation.mu_range)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.EIM_approximation = EIM_approximation
        # Declare a new matrix to store the snapshots
        self.snapshots_matrix = SnapshotsMatrix(self.EIM_approximation.parametrized_expression.space)
        # I/O
        self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
        self.greedy_selected_parameters = GreedySelectedParametersList()
        self.greedy_errors = GreedyErrorEstimatorsList()
        #
        self.offline.__func__.mu_index = 0
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    @override
    def set_xi_train(self, ntrain, enable_import=True, sampling=None):
        import_successful = ReductionMethod.set_xi_train(self, ntrain, enable_import, sampling)
        # Since exact evaluation is required, we cannot use a distributed xi_train
        self.xi_train.distributed_max = False
        return import_successful
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    @override
    def _init_offline(self):
        # Prepare folders and init EIM approximation
        all_folders = Folders()
        all_folders.update(self.folder)
        all_folders.update(self.EIM_approximation.folder)
        all_folders.pop("xi_test") # this is required only in the error analysis
        at_least_one_folder_created = all_folders.create()
        if not at_least_one_folder_created:
            self.EIM_approximation.init("online")
            return False # offline construction should be skipped, since data are already available
        else:
            self.EIM_approximation.init("offline")
            return True # offline construction should be carried out
            
    ## Finalize data structures required after the offline phase
    @override
    def _finalize_offline(self):
        self.EIM_approximation.init("online")
    
    ## Perform the offline phase of EIM
    @override
    def offline(self):
        need_to_do_offline_stage = self._init_offline()
        if not need_to_do_offline_stage:
            return self.EIM_approximation
        
        # Evaluate the parametrized expression for all parameters in xi_train
        print("==============================================================")
        print("=             EIM preprocessing phase begins                 =")
        print("==============================================================")
        print("")
        
        for run in range(len(self.xi_train)):
            print(":::::::::::::::::::::::::::::: EIM run =", run, "::::::::::::::::::::::::::::::")
            
            print("evaluate parametrized function")
            self.EIM_approximation.set_mu(self.xi_train[run])
            self.EIM_approximation.snapshot = eval(self.EIM_approximation.parametrized_expression)
            self.EIM_approximation.export_solution(self.EIM_approximation.snapshot, self.folder["snapshots"], "truth_" + str(run))
            
            print("update snapshots matrix")
            self.update_snapshots_matrix(self.EIM_approximation.snapshot)

            print("")
        
        print("==============================================================")
        print("=             EIM preprocessing phase ends                   =")
        print("==============================================================")
        print("")
        
        print("==============================================================")
        print("=             EIM offline phase begins                       =")
        print("==============================================================")
        print("")
        
        # Arbitrarily start from the first parameter in the training set
        self.EIM_approximation.set_mu(self.xi_train[0])
        self.offline.__func__.mu_index = 0
        # Resize the interpolation matrix
        self.EIM_approximation.interpolation_matrix[0] = OnlineMatrix(self.Nmax, self.Nmax)
        for run in range(self.Nmax):
            print(":::::::::::::::::::::::::::::: EIM run =", run, "::::::::::::::::::::::::::::::")
            
            print("solve eim for mu =", self.EIM_approximation.mu)
            self.EIM_approximation.solve()
            
            print("compute maximum interpolation error")
            self.EIM_approximation.snapshot = self.load_snapshot()
            (error, maximum_error, maximum_location) = self.EIM_approximation.compute_maximum_interpolation_error()
            self.update_interpolation_locations(maximum_location)
            
            print("update basis matrix")
            self.update_basis_matrix(error, maximum_error)
            
            print("update interpolation matrix")
            self.update_interpolation_matrix()
            
            if self.EIM_approximation.N < self.Nmax:
                print("find next mu")
                
            self.greedy()

            print("")
            
        print("==============================================================")
        print("=             EIM offline phase ends                         =")
        print("==============================================================")
        print("")
        
        # mu_index does not make any sense from now on
        self.offline.__func__.mu_index = None
        
        self._finalize_offline()
        return self.EIM_approximation
        
    ## Update the snapshots matrix
    def update_snapshots_matrix(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
        
    ## Update basis matrix
    def update_basis_matrix(self, error, maximum_error):
        self.EIM_approximation.Z.enrich(rescale(error, 1./maximum_error))
        self.EIM_approximation.Z.save(self.EIM_approximation.folder["basis"], "basis")
        self.EIM_approximation.N += 1
        
    def update_interpolation_locations(self, maximum_location):
        self.EIM_approximation.interpolation_locations.append(maximum_location)
        self.EIM_approximation.interpolation_locations.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_locations")
    
    ## Assemble the interpolation matrix
    def update_interpolation_matrix(self):
        last_location = self.EIM_approximation.interpolation_locations[self.EIM_approximation.N - 1]
        for j in range(self.EIM_approximation.N):
            value = eval(self.EIM_approximation.Z[j], last_location)
            self.EIM_approximation.interpolation_matrix[0][self.EIM_approximation.N - 1, j] = value
        self.EIM_approximation.interpolation_matrix.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_matrix")
            
    ## Load the precomputed snapshot
    def load_snapshot(self):
        mu = self.EIM_approximation.mu
        mu_index = self.offline.__func__.mu_index
        assert mu_index is not None
        assert mu == self.xi_train[mu_index]
        return self.snapshots_matrix[mu_index]
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        def solve_and_computer_error(mu, index):
            self.offline.__func__.mu_index = index
            self.EIM_approximation.set_mu(mu)
            
            self.EIM_approximation.solve()
            self.EIM_approximation.snapshot = self.load_snapshot()
            (_, err, _) = self.EIM_approximation.compute_maximum_interpolation_error()
            return err
            
        (error_max, error_argmax) = self.xi_train.max(solve_and_computer_error, abs)
        print("maximum EIM error =", abs(error_max))
        self.EIM_approximation.set_mu(self.xi_train[error_argmax])
        self.offline.__func__.mu_index = error_argmax
        self.greedy_selected_parameters.append(self.xi_train[error_argmax])
        self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
        self.greedy_errors.append(error_max)
        self.greedy_errors.save(self.folder["post_processing"], "error_max")
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    @override
    def _init_error_analysis(self, with_respect_to=None):
        assert with_respect_to is None
        
    @override
    def _finalize_error_analysis(self, with_respect_to=None):
        assert with_respect_to is None
    
    # Compute the error of the empirical interpolation approximation with respect to the
    # exact function over the test set
    @override
    def error_analysis(self, N=None, with_respect_to=None, **kwargs):
        if N is None:
            N = self.EIM_approximation.N
        assert with_respect_to is None # it does not makes sense to compare to something else other than the exact parametrized function
        assert len(kwargs) == 0 # not used in this method
            
        self._init_error_analysis(with_respect_to)
        
        print("==============================================================")
        print("=             EIM error analysis begins                      =")
        print("==============================================================")
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.xi_test)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("error", group_name="eim", operations="mean")
        
        for run in range(len(self.xi_test)):
            print(":::::::::::::::::::::::::::::: EIM run =", run, "::::::::::::::::::::::::::::::")
            
            self.EIM_approximation.set_mu(self.xi_test[run])
            
            # Evaluate the exact function on the truth grid
            self.EIM_approximation.snapshot = eval(self.EIM_approximation.parametrized_expression)
            
            for n in range(1, N + 1): # n = 1, ... N
                self.EIM_approximation.solve(n)
                (_, error_analysis_table["error", n, run], _) = self.EIM_approximation.compute_maximum_interpolation_error(n)
                error_analysis_table["error", n, run] = abs(error_analysis_table["error", n, run])
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print("==============================================================")
        print("=             EIM error analysis ends                        =")
        print("==============================================================")
        print("")
        
        self._finalize_error_analysis(with_respect_to)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ###########################
    
