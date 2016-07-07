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
from numpy import log, exp, mean, sqrt # for error analysis
import os # for path and makedir
import shutil # for rm
import random # to randomize selection in case of equal error bound

def EIMDecoratedReductionMethod(ReductionMethod_DerivedClass):
    class EIMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(truth_problem)
            assert isinstance(truth_problem, EIMDecoratedProblem_Class)
            # Attach EIM reduction methods
            self.EIM_reduction_method = []
            for i in range(len(truth_problem.EIM_approximation)):
                self.EIM_reduction_method.append(
                    _EIMReductionMethod(truth_problem.EIM_approximation[i], truth_problem.name() + "/eim/" + str(i))
                )
                                    
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
        
            # Propagate the values of all setters also to the EIM object
            
            ## OFFLINE: set maximum reduced space dimension (stopping criterion)
            def set_Nmax(self, Nmax, **kwargs):
                ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
                assert "Nmax_EIM" in kwargs
                Nmax_EIM = kwargs["Nmax_EIM"]
                if isinstance(nmax_EIM, tuple):
                    assert len(nmax_EIM) = len(self.EIM_reduction_method)
                    for i in range(len(self.EIM_reduction_method)):
                        self.EIM_reduction_method[i].set_Nmax(Nmax_EIM[i]) # kwargs are not needed
                else:
                    assert isinstance(nmax_EIM, int)
                    for i in range(len(self.EIM_reduction_method)):
                        self.EIM_reduction_method[i].set_Nmax(Nmax_EIM) # kwargs are not needed

                
            ## OFFLINE: set the elements in the training set \xi_train.
            def set_xi_train(self, ntrain, enable_import=True, sampling="random"):
                EllipticCoerciveRBBase.set_xi_train(self, ntrain, enable_import, sampling)
                for i in range(len(self.EIM_reduction_method)):
                    self.EIM_reduction_method[i].set_xi_train(ntrain, enable_import, sampling)
                
            ## ERROR ANALYSIS: set the elements in the test set \xi_test.
            def set_xi_test(self, ntest, enable_import=False, sampling="random"):
                EllipticCoerciveRBBase.set_xi_test(self, ntest, enable_import, sampling)
                for i in range(len(self.EIM_reduction_method)):
                    self.EIM_reduction_method[i].set_xi_test(ntest, enable_import, sampling)
                
            #  @}
            ########################### end - SETTERS - end ########################### 
            
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
        
            ## Perform the offline phase of the reduced order model
            def offline(self):
                # Perform first the EIM offline phase, ...
                bak_first_mu = tuple(list(self.truth_problem.mu))
                for i in range(len(self.EIM_reduction_method)):
                    self.EIM_reduction_method[i].offline()
                # ..., and then call the parent method.
                self.truth_problem.set_mu(bak_first_mu)
                ReductionMethod_DerivedClass.offline(self)
        
            #  @}
            ########################### end - OFFLINE STAGE - end ###########################
        
            ###########################     ERROR ANALYSIS     ########################### 
            ## @defgroup ErrorAnalysis Error analysis
            #  @{
        
            # Compute the error of the reduced order approximation with respect to the full order one
            # over the test set
            def error_analysis(self, N=None):
                # Perform first the EIM error analysis, ...
                for i in range(len(self.EIM_reduction_method)):
                    self.EIM_reduction_method[i].error_analysis(N)
                # ..., and then call the parent method.
                ReductionMethod_DerivedClass.error_analysis(self, N)        
                
            #  @}
            ########################### end - ERROR ANALYSIS - end ########################### 
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class EIM
    #
    # Empirical interpolation method for the interpolation of parametrized functions
    class _EIMReductionMethod(ReductionMethodBase):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the EIM object
        #  @{
        
        ## Default initialization of members
        def __init__(self, EIM_approximation, folder_prefix):
            # Call the parent initialization
            ReductionMethodBase.__init__(self, folder_prefix)
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # High fidelity problem
            self.EIM_approximation = EIM_approximation
            # Store the dof to vertex map to locate maximum of functions
            self.dof_to_vertex_map = dof_to_vertex_map(self.EIM_approximation.V)
            # 6bis. Declare a new matrix to store the snapshots
            self.snapshot_matrix = SnapshotMatrix()
            # 9. I/O
            self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
            self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
            #
            self.mu_index = 0
            
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Initialize data structures required for the offline phase
        def _init_offline(self):
            # Prepare folders and init reduced problem
            all_folders_exist = True
            for f in self.folder.values():
                if not os.path.exists(f):
                    all_folders_exist = False
                    os.makedirs(f)
            if all_folders_exist:
                self.reduced_problem.init("online")
                return False # offline construction should be skipped, since data are already available
            else:
                self.reduced_problem.init("offline")
                return True # offline construction should be carried out
        
        ## Perform the offline phase of EIM
        def offline(self):
            need_to_do_offline_stage = self._init_offline()
            if not need_to_do_offline_stage:
                return self.EIM_approximation
                
            # Interpolate the parametrized function on the mesh grid for all parameters in xi_train
            print("==============================================================")
            print("=             EIM preprocessing phase begins                 =")
            print("==============================================================")
            print("")
            
            for run in range(len(self.xi_train)):
                print(":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::")
                
                print("evaluate parametrized function")
                self.EIM_approximation.set_mu(self.xi_train[run])
                f = self.EIM_approximation.evaluate_parametrized_function_at_mu(self.EIM_approximation.mu)
                snapshot = interpolate(f, self.V)
                self.EIM_approximation.export_solution(self.snapshot, self.folder["snapshots"], "truth_" + str(run))
                
                print("update snapshot matrix")
                self.update_snapshot_matrix(snapshot)

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
            self.set_mu(self.xi_train[0])
            self.mu_index = 0
            # Resize the interpolation matrix
            self.EIM_approximation.interpolation_matrix = OnlineMatrix(self.Nmax, self.Nmax)            
            for run in range(self.Nmax):
                print(":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::")
                
                print("solve eim for mu = ", self.mu)
                self.EIM_approximation.solve()
                
                print("compute maximum interpolation error")
                # TODO move into a method
                (error, maximum_error, maximum_point, maximum_point_dof) = self.compute_maximum_interpolation_error(output_error=True, output location=True)
                self.EIM_approximation.interpolation_points.append(maximum_point)
                self.EIM_approximation.interpolation_points.save(self.folder["reduced_operators"], "interpolation_points")
                self.EIM_approximation.interpolation_points_dof.append(maximum_point_dof)
                self.EIM_approximation.interpolation_points_dof.save(self.folder["reduced_operators"], "interpolation_points_dof")
                
                print("update basis matrix")
                # TODO move into the update_basis_matrix method
                error /= maximum_error
                self.EIM_approximation.Z.enrich(error)
                self.EIM_approximation.Z.save(self.EIM_approximation.folder["basis"], "basis")
                self.EIM_approximation.N += 1
                
                print("update interpolation matrix")
                self.EIM_approximation.update_interpolation_matrix()
                
                if self.EIM_approximation.N < self.Nmax:
                    print("find next mu")
                    self.greedy()
                else:
                    self.greedy()

                print("")
                
            print("==============================================================")
            print("=             EIM offline phase ends                         =")
            print("==============================================================")
            print("")
            
            # mu_index does not make any sense from now on
            self.mu_index = None
            
        ## Update the snapshot matrix
        def update_snapshot_matrix(self, snapshot):
            self.snapshot_matrix.append(snapshot)
                
        ## Load the precomputed snapshot
        def load_snapshot(self):
            mu = self.mu
            mu_index = self.mu_index
            if mu != self.xi_train[mu_index]:
                # There is something wrong if we are here...
                raise RuntimeError("Should never arrive here")
            return self.snapshot_matrix[mu_index]
        
        # Compute the interpolation error and/or its maximum location
        def compute_maximum_interpolation_error(self, N=None, **output_options):
            if N is None:
                N = self.N
            if not "output_error" in output_options:
                output_options["output_error"] = False
            if not "output_location" in output_options:
                output_options["output_location"] = False
            
            # Compute the error (difference with the eim approximation)
            if N > 0:
                snapshot_EIM = self.EIM_approximation.Z*self.EIM_approximation._interpolation_coefficients
                snapshot_EIM -= self.load_snapshot()
                error = snapshot_EIM # error as a function
            else:
                error = self.load_snapshot()
            
            if output_options["output_error"] and not output_options["output_location"]:
                maximum_error = error.vector().norm("linf")
            elif output_options["output_location"]:
                # Locate the vertex of the mesh where the error is maximum
                maximum_error = 0.0
                maximum_point = None
                maximum_point_dof = None
                for dof_index in range(self.V.dim()):
                    vertex_index = self.dof_to_vertex_map[dof_index]
                    err = error.vector()[dof_index]
                    if (abs(err) > abs(maximum_error) or (abs(err) == abs(maximum_error) and random.random() >= 0.5)):
                        maximum_error = err
                        maximum_point = self.V.mesh().coordinates()[vertex_index]
                        maximum_point_dof = dof_index
            else:
                raise RuntimeError("Invalid output options")
                
            # Return
            if output_options["output_error"] and output_options["output_location"]:
                return (error, abs(maximum_error), maximum_point, maximum_point_dof)
            elif output_options["output_error"]:
                return (error, abs(maximum_error))
            elif output_options["output_location"]:
                return (maximum_point, maximum_point_dof)
            else:
                raise RuntimeError("Invalid output options")
                                
        ## Choose the next parameter in the offline stage in a greedy fashion
        def greedy(self):
            err_max = -1.0
            munew = None
            munew_index = None
            for i in range(len(self.xi_train)):
                self.EIM_approximation.set_mu(self.xi_train[i])
                self.mu_index = i
                
                # Compute the EIM approximation ...
                self.EIM_approximation.solve()
                
                # ... and compute the maximum error
                (_, err) = self.compute_maximum_interpolation_error(output_error=True)
                
                if (err > err_max):
                    err_max = err
                    munew = self.xi_train[i]
                    munew_index = i
            assert err_max > 0.
            assert munew is not None
            assert munew_index is not None
            print("absolute error max = ", err_max)
            self.EIM_approximation.set_mu(munew)
            self.mu_index = munew_index
            self.save_greedy_post_processing_file(self.N, err_max, munew, self.folder["post_processing"])
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
        
        # Compute the error of the empirical interpolation approximation with respect to the
        # exact function over the test set
        def error_analysis(self, N=None):
            self.init()
            if N is None:
                N = self.N
                
            print("==============================================================")
            print("=             EIM error analysis begins                      =")
            print("==============================================================")
            print("")
            
            error = np.zeros((N, len(self.xi_test)))
            
            for run in range(len(self.xi_test)):
                print(":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::")
                
                self.set_mu(self.xi_test[run])
                
                # Evaluate the exact function on the truth grid
                f = self.evaluate_parametrized_function_at_mu(self.mu)
                self.snapshot = interpolate(f, self.V)
                
                for n in range(N): # n = 0, 1, ... N - 1
                    self.online_solve(n)
                    error[n, run] = self.compute_maximum_interpolation_error(n, output_error=True)
            
            # Print some statistics
            print("")
            print("N \t gmean(err)")
            for n in range(N): # n = 0, 1, ... N - 1
                mean_error = np.exp(np.mean(np.log((error[n, :]))))
                print(str(n+1) + " \t " + str(mean_error))
            
            print("")
            print("==============================================================")
            print("=             EIM error analysis ends                        =")
            print("==============================================================")
            print("")
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
        ###########################     I/O     ########################### 
        ## @defgroup IO Input/output methods
        #  @{
    
        ## Save greedy post processing to file
        @staticmethod
        def save_greedy_post_processing_file(N, err_max, mu_greedy, directory):
            with open(directory + "/error_max.txt", "a") as outfile:
                file.write(str(N) + " " + str(err_max))
            with open(directory + "/mu_greedy.txt", "a") as outfile:
                file.write(str(mu_greedy))
            
        #  @}
        ########################### end - I/O - end ########################### 
        
    # return value (a class) for the decorator
    return EIMDecoratedReductionMethod_Class
    
