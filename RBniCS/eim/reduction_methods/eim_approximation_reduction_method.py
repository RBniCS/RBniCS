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
from RBniCS.backends import evaluate, rescale
from RBniCS.backends.online import OnlineMatrix
from RBniCS.utils.io import Folders, ErrorAnalysisTable, SpeedupAnalysisTable, GreedySelectedParametersList, GreedyErrorEstimatorsList
from RBniCS.utils.mpi import print
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
    def __init__(self, EIM_approximation):
        # Call the parent initialization
        ReductionMethod.__init__(self, EIM_approximation.folder_prefix, EIM_approximation.mu_range)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.EIM_approximation = EIM_approximation
        # Declare a new container to store the snapshots
        self.snapshots_container = self.EIM_approximation.parametrized_expression.create_snapshots_container()
        # I/O
        self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
        self.greedy_selected_parameters = GreedySelectedParametersList()
        self.greedy_errors = GreedyErrorEstimatorsList()
        #
        self._offline__mu_index = 0
        
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
            
        interpolation_method_name = self.EIM_approximation.parametrized_expression.interpolation_method_name()
        interpolation_method_name_headings = interpolation_method_name.rjust(4)
        
        # Evaluate the parametrized expression for all parameters in xi_train
        print("==============================================================")
        print("=            " + interpolation_method_name_headings + " preprocessing phase begins                 =")
        print("==============================================================")
        print("")
        
        for (run, mu) in enumerate(self.xi_train):
            print(":::::::::::::::::::::::::::::: " + interpolation_method_name + " run =", run, "::::::::::::::::::::::::::::::")
            
            print("evaluate parametrized expression")
            self.EIM_approximation.set_mu(mu)
            self.EIM_approximation.snapshot = evaluate(self.EIM_approximation.parametrized_expression)
            self.EIM_approximation.export_solution(self.EIM_approximation.snapshot, self.folder["snapshots"], "truth_" + str(run))
            
            print("add to snapshots")
            self.add_to_snapshots(self.EIM_approximation.snapshot)

            print("")
            
        # If basis generation is POD, compute the first Nmax POD modes of the snapshots
        if self.EIM_approximation.basis_generation == "POD":
            print("compute basis")
            self.compute_basis_POD()
            print("")
        
        print("==============================================================")
        print("=            " + interpolation_method_name_headings + " preprocessing phase ends                   =")
        print("==============================================================")
        print("")
        
        print("==============================================================")
        print("=            " + interpolation_method_name_headings + " offline phase begins                       =")
        print("==============================================================")
        print("")
        
        # Arbitrarily start from the first parameter in the training set (Greedy only)
        if self.EIM_approximation.basis_generation == "Greedy":
            self.EIM_approximation.set_mu(self.xi_train[0])
            self._offline__mu_index = 0
        # Resize the interpolation matrix
        self.EIM_approximation.interpolation_matrix[0] = OnlineMatrix(self.Nmax, self.Nmax)
        for run in range(self.Nmax):
            print(":::::::::::::::::::::::::::::: " + interpolation_method_name + " run =", run, "::::::::::::::::::::::::::::::")
            
            if self.EIM_approximation.basis_generation == "Greedy":
                print("solve interpolation for mu =", self.EIM_approximation.mu)
                self.EIM_approximation.solve()
                
                print("compute and locate maximum interpolation error")
                self.EIM_approximation.snapshot = self.load_snapshot()
                (error, maximum_error, maximum_location) = self.EIM_approximation.compute_maximum_interpolation_error()
                
                print("update locations with", maximum_location)
                self.update_interpolation_locations(maximum_location)
                
                print("update basis")
                self.update_basis_greedy(error, maximum_error)
                
                print("update interpolation matrix")
                self.update_interpolation_matrix()
                
                if self.EIM_approximation.N < self.Nmax:
                    print("find next mu")
                    
                self.greedy()
                
            else:
                print("solve interpolation for basis number", self.EIM_approximation.N)
                self.EIM_approximation._solve(self.EIM_approximation.Z[self.EIM_approximation.N])
                
                print("compute and locate maximum interpolation error")
                self.EIM_approximation.snapshot = self.EIM_approximation.Z[self.EIM_approximation.N]
                (error, maximum_error, maximum_location) = self.EIM_approximation.compute_maximum_interpolation_error()
                
                print("update locations with", maximum_location)
                self.update_interpolation_locations(maximum_location)
                
                self.EIM_approximation.N += 1
                
                print("update interpolation matrix")
                self.update_interpolation_matrix()
                
            print("")
            
        print("==============================================================")
        print("=            " + interpolation_method_name_headings + " offline phase ends                         =")
        print("==============================================================")
        print("")
        
        # mu_index does not make any sense from now on (Greedy only)
        if self.EIM_approximation.basis_generation == "Greedy":
            self._offline__mu_index = None
        
        self._finalize_offline()
        return self.EIM_approximation
        
    ## Update the snapshots container
    def add_to_snapshots(self, snapshot):
        self.snapshots_container.enrich(snapshot)
        
    ## Update basis (greedy version)
    def update_basis_greedy(self, error, maximum_error):
        self.EIM_approximation.Z.enrich(rescale(error, 1./maximum_error))
        self.EIM_approximation.Z.save(self.EIM_approximation.folder["basis"], "basis")
        self.EIM_approximation.N += 1

    ## Update basis (POD version)
    def compute_basis_POD(self):
        POD = self.EIM_approximation.parametrized_expression.create_POD_container()
        POD.store_snapshot(self.snapshots_container)
        (Z, N) = POD.apply(self.Nmax)
        self.EIM_approximation.Z.enrich(Z)
        self.EIM_approximation.Z.save(self.EIM_approximation.folder["basis"], "basis")
        # do not increment self.EIM_approximation.N
        POD.print_eigenvalues(N)
        POD.save_eigenvalues_file(self.folder["post_processing"], "eigs")
        POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
        
    def update_interpolation_locations(self, maximum_location):
        self.EIM_approximation.interpolation_locations.append(maximum_location)
        self.EIM_approximation.interpolation_locations.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_locations")
    
    ## Assemble the interpolation matrix
    def update_interpolation_matrix(self):
        self.EIM_approximation.interpolation_matrix[0] = evaluate(self.EIM_approximation.Z[:self.EIM_approximation.N], self.EIM_approximation.interpolation_locations)
        self.EIM_approximation.interpolation_matrix.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_matrix")
            
    ## Load the precomputed snapshot
    def load_snapshot(self):
        assert self.EIM_approximation.basis_generation == "Greedy"
        mu = self.EIM_approximation.mu
        mu_index = self._offline__mu_index
        assert mu_index is not None
        assert mu == self.xi_train[mu_index]
        return self.snapshots_container[mu_index]
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        assert self.EIM_approximation.basis_generation == "Greedy"
        def solve_and_computer_error(mu, index):
            self._offline__mu_index = index
            self.EIM_approximation.set_mu(mu)
            
            self.EIM_approximation.solve()
            self.EIM_approximation.snapshot = self.load_snapshot()
            (_, err, _) = self.EIM_approximation.compute_maximum_interpolation_error()
            return err
            
        (error_max, error_argmax) = self.xi_train.max(solve_and_computer_error, abs)
        print("maximum interpolation error =", abs(error_max))
        self.EIM_approximation.set_mu(self.xi_train[error_argmax])
        self._offline__mu_index = error_argmax
        self.greedy_selected_parameters.append(self.xi_train[error_argmax])
        self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
        self.greedy_errors.append(error_max)
        self.greedy_errors.save(self.folder["post_processing"], "error_max")
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the empirical interpolation approximation with respect to the
    # exact function over the test set
    @override
    def error_analysis(self, N=None, **kwargs):
        if N is None:
            N = self.EIM_approximation.N
        assert len(kwargs) == 0 # not used in this method
            
        self._init_error_analysis(**kwargs)
        
        interpolation_method_name = self.EIM_approximation.parametrized_expression.interpolation_method_name()
        interpolation_method_name_headings = interpolation_method_name.rjust(4)
        
        print("==============================================================")
        print("=            " + interpolation_method_name_headings + " error analysis begins                      =")
        print("==============================================================")
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.xi_test)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("error", group_name="eim", operations="mean")
        
        for (run, mu) in enumerate(self.xi_test):
            print(":::::::::::::::::::::::::::::: " + interpolation_method_name + " run =", run, "::::::::::::::::::::::::::::::")
            
            self.EIM_approximation.set_mu(mu)
            
            # Evaluate the exact function on the truth grid
            self.EIM_approximation.snapshot = evaluate(self.EIM_approximation.parametrized_expression)
            
            for n in range(1, N + 1): # n = 1, ... N
                self.EIM_approximation.solve(n)
                (_, error_analysis_table["error", n, run], _) = self.EIM_approximation.compute_maximum_interpolation_error(n)
                error_analysis_table["error", n, run] = abs(error_analysis_table["error", n, run])
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print("==============================================================")
        print("=            " + interpolation_method_name_headings + " error analysis ends                        =")
        print("==============================================================")
        print("")
        
        self._finalize_error_analysis(**kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ###########################
    
