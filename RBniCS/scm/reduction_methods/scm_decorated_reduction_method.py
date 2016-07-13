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
## @file scm.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os
import random # to randomize selection in case of equal error bound
from RBniCS.reduction_methods import ReductionMethod
from RBniCS.scm.problems.parametrized_hermitian_eigenproblem import ParametrizedHermitianEigenProblem

def SCMDecoratedReductionMethod(ReductionMethod_DerivedClass):

    #~~~~~~~~~~~~~~~~~~~~~~~~~     SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class SCM
    #
    # Empirical interpolation method for the interpolation of parametrized functions
    class _SCMReductionMethod(ReductionMethod):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the SCM object
        #  @{
        
        ## Default initialization of members
        def __init__(self, SCM_approximation, folder_prefix):
            # Call the parent initialization
            ReductionMethod.__init__(self, folder_prefix, SCM_approximation.mu_range)
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # High fidelity problem
            self.SCM_approximation = SCM_approximation
            # I/O
            self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
            self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
            #
            self.offline.__func__.mu_index = 0
            
            # Get data that were temporarily store in the SCM_approximation
            self.constrain_minimum_eigenvalue = self.SCM_approximation._input_storage_for_SCM_reduction.constrain_minimum_eigenvalue
            self.constrain_maximum_eigenvalue = self.SCM_approximation._input_storage_for_SCM_reduction.constrain_minimum_eigenvalue
            self.bounding_box_minimum_eigensolver_parameters = self.SCM_approximation._input_storage_for_SCM_reduction.bounding_box_minimum_eigensolver_parameters
            self.bounding_box_maximum_eigensolver_parameters = self.SCM_approximation._input_storage_for_SCM_reduction.bounding_box_maximum_eigensolver_parameters
            del self.SCM_approximation._input_storage_for_SCM_reduction
            
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Initialize data structures required for the offline phase
        def _init_offline(self):
            # Prepare folders and init SCM approximation
            all_folders_exist = True
            all_folders = list()
            all_folders.extend(self.folder.values())
            all_folders.extend(self.SCM_approximation.folder.values())
            for f in all_folders:
                if os.path.exists(f) and len(os.listdir(f)) == 0: # already created, but empty
                    all_folders_exist = False
                if not os.path.exists(f):
                    all_folders_exist = False
                    os.makedirs(f)
            if all_folders_exist:
                self.SCM_approximation.init("online")
                return False # offline construction should be skipped, since data are already available
            else:
                self.SCM_approximation.init("offline")
                return True # offline construction should be carried out
        
        ## Perform the offline phase of SCM
        def offline(self):
            need_to_do_offline_stage = self._init_offline()
            if not need_to_do_offline_stage:
                return self.SCM_approximation
            
            print("==============================================================")
            print("=             SCM offline phase begins                       =")
            print("==============================================================")
            print("")
            
            # Compute the bounding box \mathcal{B}
            self.compute_bounding_box()
            
            # Arbitrarily start from the first parameter in the training set
            self.set_mu(self.xi_train[0])
            self.offline.__func__.mu_index = 0
            
            for run in range(self.Nmax):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run = ", run, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                
                # Store the greedy parameter
                self.update_C_J()
                
                # Evaluate the coercivity constant
                print("evaluate the stability factor for mu = ", self.mu)
                (alpha, eigenvector) = self.SCM_approximation.exact_coercivity_constant_calculator.solve()
                
                # Update internal data structures
                self.update_alpha_J(alpha)
                self.update_eigenvector_J(eigenvector)
                UB_vector = self.compute_UB_vector(eigenvector)
                self.update_UB_vectors_J(UB_vector)
                                
                # Prepare for next iteration
                if self.N < self.Nmax:
                    print("find next mu")
                    
                self.greedy()
            
            print("==============================================================")
            print("=             SCM offline phase ends                         =")
            print("==============================================================")
            print("")
            
            # mu_index does not make any sense from now on
            self.offline.__func__.mu_index = None
            
            self.SCM_approximation.init("online")
            return self.SCM_approximation
            
        # Compute the bounding box \mathcal{B}
        def compute_bounding_box(self):
            # Resize the bounding box storage
            Q = self.truth_problem.Q["a"]
            
            for q in range(Q):
                # Compute the minimum eigenvalue
                minimum_eigenvalue_calculator = ParametrizedHermitianEigenProblem(self.truth_problem, ("a", q), False, self.constrain_minimum_eigenvalue, "smallest", self.bounding_box_minimum_eigensolver_parameters)
                (self.B_min[q], _) = minimum_eigenvalue_calculator.solve()
                print("B_min[" + str(q) + "] = " + str(self.B_min[q]))
                
                # Compute the maximum eigenvalue
                maximum_eigenvalue_calculator = ParametrizedHermitianEigenProblem(self.truth_problem, ("a", q), False, self.constrain_maximum_eigenvalue, "largest", self.bounding_box_maximum_eigensolver_parameters)
                (self.B_max[q], _) = maximum_eigenvalue_calculator.solve()
                print("B_max[" + str(q) + "] = " + str(self.B_max[q]))
            
            # Save to file
            self.B_min.save(self.folder["reduced_operators"], "B_min")
            self.B_max.save(self.folder["reduced_operators"], "B_max")
            
        # Store the greedy parameter
        def update_C_J(self):
            mu = self.SCM_approximation.mu
            mu_index = self.offline.__func__.mu_index
            assert mu == self.xi_train[mu_index]
            
            self.C_J.append(mu_index)
            self.N = len(self.C_J)
            
            if mu_index in self.complement_C_J: # if not SCM selects twice the same parameter
                self.complement_C_J.remove(mu_index)
            
            # Save to file
            self.C_J.save(self.folder["reduced_operators"], "C_J")
            self.complement_C_J.save(self.folder["reduced_operators"], "complement_C_J")
            
        def update_alpha_J(self, alpha):
            self.alpha_J.append(alpha)
            self.alpha_J.save(self.folder["reduced_operators"], "alpha_J")
            
        def update_eigenvector_J(self, eigenvector):
            self.eigenvector_J.append(eigenvector)
            self.export_solution(eigenvector, self.folder["snapshots"], "eigenvector_" + str(len(self.eigenvector_J) - 1))
            
        ## Compute the ratio between a_q(u,u) and s(u,u), for all q in vec
        def compute_UB_vector(self, u):
            Q = self.truth_problem.Q["a"]
            X = self.truth_problem.inner_product[0]
            UB_vector = OnlineVector(Q)
            norm_S_squared = transpose(u)*X*u
            for q in range(Q):
                A_q = self.truth_problem.operator["a"][q]
                UB_vector[q] = (transpose(u)*A_q*u)/norm_S_squared
            return UB_vector
            
        def update_UB_vectors_J(self, UB_vector):
            self.UB_vectors_J.append(UB_vector)
            self.UB_vectors_J.save(self.folder["reduced_operators"], "UB_vectors_J")
            
        ## Choose the next parameter in the offline stage in a greedy fashion
        def greedy(self):
            ntrain = len(self.xi_train)
            alpha_LB_on_xi_train = CoercivityConstantsList(ntrain)
            #
            delta_max = -1.0
            munew = None
            munew_index = None
            for i in range(ntrain):
                mu = self.xi_train[i]
                self.offline.__func__.mu_index = i
                self.set_mu(mu)
                LB = self.get_alpha_LB(mu, False)
                UB = self.get_alpha_UB(mu)
                delta = (UB - LB)/UB

                if LB/UB < 0:
                    print("SCM warning at mu = ", mu , ": LB = ", LB, " < 0")
                if LB/UB > 1:
                    print("SCM warning at mu = ", mu , ": LB = ", LB, " > UB = ", UB)
                    
                alpha_LB_on_xi_train[i] = max(0, LB)
                if ((delta > delta_max) or (delta == delta_max and random.random() >= 0.5)):
                    delta_max = delta
                    munew = mu
                    munew_index = i
            assert delta_max > 0.
            assert munew is not None
            assert munew_index is not None
            print("absolute SCM delta max = ", delta_max)
            self.SCM_approximation.set_mu(munew)
            self.offline.__func__.mu_index = munew_index
            self.save_greedy_post_processing_file(self.SCM_approximation.N, err_max, munew, self.folder["post_processing"])
            
            # Overwrite alpha_LB_on_xi_train
            self.SCM_approximation.alpha_LB_on_xi_train = alpha_LB_on_xi_train
            self.SCM_approximation.alpha_LB_on_xi_train.save(self.SCM_approximation.folder["reduced_operators"], "alpha_LB_on_xi_train")
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
        
        # Compute the error of the empirical interpolation approximation with respect to the
        # exact function over the test set
        def error_analysis(self, N=None):
            if N is None:
                N = self.SCM_approximation.N
                
            print("==============================================================")
            print("=             SCM error analysis begins                      =")
            print("==============================================================")
            print("")
            
            error_analysis_table = ErrorAnalysisTable(self.xi_test)
            error_analysis_table.set_Nmin(N)
            error_analysis_table.set_Nmax(N)
            error_analysis_table.add_column("normalized_error", group_name="scm", operations=("min", "mean", "max"))
            
            for run in range(len(self.xi_test)):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run = ", run, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                
                self.SCM_approximation.set_mu(self.xi_test[run])
                
                # Truth solve
                (alpha, _) = self.SCM_approximation.exact_coercivity_constant_calculator.solve()
                
                alpha_LB = self.get_alpha_LB(self.mu, False)
                alpha_UB = self.get_alpha_UB(self.mu)
                if alpha_LB/alpha_UB < 0:
                    print("SCM warning at mu = ", self.mu , ": LB = ", alpha_LB, " < 0")
                if alpha_LB/alpha_UB > 1:
                    print("SCM warning at mu = ", self.mu , ": LB = ", alpha_LB, " > UB = ", alpha_UB)
                if alpha_LB/alpha > 1:
                    print("SCM warning at mu = ", self.mu , ": LB = ", alpha_LB, " > exact = ", alpha)
                
                error_analysis_table["normalized_error", N, run] = (alpha - alpha_LB)/alpha_UB
            
            # Print
            print("")
            print(error_analysis_table)
            
            print("")
            print("==============================================================")
            print("=             SCM error analysis ends                        =")
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
                outfile.write(str(N) + " " + str(err_max) + "\n")
            with open(directory + "/mu_greedy.txt", "a") as outfile:
                outfile.write(str(mu_greedy) + "\n")
            
        #  @}
        ########################### end - I/O - end ########################### 

    class SCMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(self, truth_problem)
            
            # Storage for SCM reduction method
            self.SCM_reduction = _SCMReduction(self.truth_problem.SCM_approximation, self.truth_problem.name() + "/scm")
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the SCM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_Nmax(self, Nmax, **kwargs):
            ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "SCM" in kwargs
            Nmax_SCM = kwargs["SCM"]
            assert isinstance(Nmax_SCM, int)
            self.SCM_reduction.set_Nmax(Nmax_SCM) # kwargs are not needed

            
        ## OFFLINE: set the elements in the training set \xi_train.
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            ReductionMethod_DerivedClass.set_xi_train(self, ntrain, enable_import, sampling)
            import_successful = self.SCM_reduction.set_xi_train(ntrain, enable_import=True, sampling)
            assert import_successful == True
            
        ## ERROR ANALYSIS: set the elements in the test set \xi_test.
        def set_xi_test(self, ntest, enable_import=False, sampling=None):
            ReductionMethod_DerivedClass.set_xi_test(self, ntest, enable_import, sampling)
            self.SCM_reduction.set_xi_test(ntest, enable_import, sampling)
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
    
        ## Perform the offline phase of the reduced order model
        def offline(self):
            # Perform first the SCM offline phase, ...
            bak_first_mu = tuple(list(self.truth_problem.mu))
            self.SCM_reduction.offline()
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
            # Perform first the SCM error analysis, ...
            self.SCM_reduction.error_analysis(N)
            # ..., and then call the parent method.
            ReductionMethod_DerivedClass.error_analysis(self, N)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return SCMDecoratedReductionMethod_Class
    
