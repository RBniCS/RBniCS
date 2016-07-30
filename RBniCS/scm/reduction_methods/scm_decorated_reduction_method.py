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
from dolfin import Function
from RBniCS.linear_algebra import transpose, OnlineVector
from RBniCS.reduction_methods import ReductionMethod
from RBniCS.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable, Folders, GreedySelectedParametersList, GreedyErrorEstimatorsList 
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import extends, override
from RBniCS.scm.problems.parametrized_hermitian_eigenproblem import ParametrizedHermitianEigenProblem

def SCMDecoratedReductionMethod(ReductionMethod_DerivedClass):

    #~~~~~~~~~~~~~~~~~~~~~~~~~     SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class SCM
    #
    # Empirical interpolation method for the interpolation of parametrized functions
    @extends(ReductionMethod)
    class _SCMReductionMethod(ReductionMethod):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the SCM object
        #  @{
        
        ## Default initialization of members
        @override
        def __init__(self, SCM_approximation, folder_prefix):
            # Call the parent initialization
            ReductionMethod.__init__(self, folder_prefix, SCM_approximation.mu_range)
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # High fidelity problem
            self.SCM_approximation = SCM_approximation
            # I/O
            self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
            self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
            self.greedy_selected_parameters = GreedySelectedParametersList()
            self.greedy_error_estimators = GreedyErrorEstimatorsList()
            #
            self.offline.__func__.mu_index = 0
            
            # Get data that were temporarily store in the SCM_approximation
            self.constrain_minimum_eigenvalue = self.SCM_approximation._input_storage_for_SCM_reduction["constrain_minimum_eigenvalue"]
            self.constrain_maximum_eigenvalue = self.SCM_approximation._input_storage_for_SCM_reduction["constrain_maximum_eigenvalue"]
            self.bounding_box_minimum_eigensolver_parameters = self.SCM_approximation._input_storage_for_SCM_reduction["bounding_box_minimum_eigensolver_parameters"]
            self.bounding_box_maximum_eigensolver_parameters = self.SCM_approximation._input_storage_for_SCM_reduction["bounding_box_maximum_eigensolver_parameters"]
            del self.SCM_approximation._input_storage_for_SCM_reduction
            
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        ## OFFLINE: set the elements in the training set \xi_train.
        @override
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod.set_xi_train(self, ntrain, enable_import, sampling)
            self.SCM_approximation.xi_train = self.xi_train
            return import_successful
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Initialize data structures required for the offline phase
        @override
        def _init_offline(self):
            # Prepare folders and init SCM approximation
            all_folders = Folders()
            all_folders.update(self.folder)
            all_folders.update(self.SCM_approximation.folder)
            all_folders.pop("xi_test") # this is required only in the error analysis
            at_least_one_folder_created = all_folders.create()
            if not at_least_one_folder_created:
                self.SCM_approximation.init("online")
                return False # offline construction should be skipped, since data are already available
            else:
                self.SCM_approximation.init("offline")
                return True # offline construction should be carried out
        
        ## Perform the offline phase of SCM
        @override
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
            self.SCM_approximation.set_mu(self.xi_train[0])
            self.offline.__func__.mu_index = 0
            
            for run in range(self.Nmax):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run =", run, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                
                # Store the greedy parameter
                self.update_C_J()
                
                # Evaluate the coercivity constant
                print("evaluate the stability factor for mu =", self.SCM_approximation.mu)
                (alpha, eigenvector) = self.SCM_approximation.exact_coercivity_constant_calculator.solve()
                
                # Update internal data structures
                self.update_alpha_J(alpha)
                self.update_eigenvector_J(eigenvector)
                UB_vector = self.compute_UB_vector(eigenvector)
                self.update_UB_vectors_J(UB_vector)
                                
                # Prepare for next iteration
                if self.SCM_approximation.N < self.Nmax:
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
            Q = self.SCM_approximation.truth_problem.Q["a"]
            
            for q in range(Q):
                # Compute the minimum eigenvalue
                minimum_eigenvalue_calculator = ParametrizedHermitianEigenProblem(self.SCM_approximation.truth_problem, ("a", q), False, self.constrain_minimum_eigenvalue, "smallest", self.bounding_box_minimum_eigensolver_parameters)
                minimum_eigenvalue_calculator.init()
                (self.SCM_approximation.B_min[q], _) = minimum_eigenvalue_calculator.solve()
                print("B_min[" + str(q) + "] = " + str(self.SCM_approximation.B_min[q]))
                
                # Compute the maximum eigenvalue
                maximum_eigenvalue_calculator = ParametrizedHermitianEigenProblem(self.SCM_approximation.truth_problem, ("a", q), False, self.constrain_maximum_eigenvalue, "largest", self.bounding_box_maximum_eigensolver_parameters)
                maximum_eigenvalue_calculator.init()
                (self.SCM_approximation.B_max[q], _) = maximum_eigenvalue_calculator.solve()
                print("B_max[" + str(q) + "] = " + str(self.SCM_approximation.B_max[q]))
            
            # Save to file
            self.SCM_approximation.B_min.save(self.SCM_approximation.folder["reduced_operators"], "B_min")
            self.SCM_approximation.B_max.save(self.SCM_approximation.folder["reduced_operators"], "B_max")
            
        # Store the greedy parameter
        def update_C_J(self):
            mu = self.SCM_approximation.mu
            mu_index = self.offline.__func__.mu_index
            assert mu == self.xi_train[mu_index]
            
            self.SCM_approximation.C_J.append(mu_index)
            self.SCM_approximation.N = len(self.SCM_approximation.C_J)
            
            if mu_index in self.SCM_approximation.complement_C_J: # if not SCM selects twice the same parameter
                self.SCM_approximation.complement_C_J.remove(mu_index)
            
            # Save to file
            self.SCM_approximation.C_J.save(self.SCM_approximation.folder["reduced_operators"], "C_J")
            self.SCM_approximation.complement_C_J.save(self.SCM_approximation.folder["reduced_operators"], "complement_C_J")
            
        def update_alpha_J(self, alpha):
            self.SCM_approximation.alpha_J.append(alpha)
            self.SCM_approximation.alpha_J.save(self.SCM_approximation.folder["reduced_operators"], "alpha_J")
            
        def update_eigenvector_J(self, eigenvector):
            self.SCM_approximation.eigenvector_J.append(eigenvector)
            eigenvector_function = Function(self.SCM_approximation.truth_problem.V, eigenvector)
            self.SCM_approximation.export_solution(eigenvector_function, self.folder["snapshots"], "eigenvector_" + str(len(self.SCM_approximation.eigenvector_J) - 1))
            
        ## Compute the ratio between a_q(u,u) and s(u,u), for all q in vec
        def compute_UB_vector(self, u):
            Q = self.SCM_approximation.truth_problem.Q["a"]
            X = self.SCM_approximation.truth_problem.inner_product[0]
            UB_vector = OnlineVector(Q)
            norm_S_squared = transpose(u)*X*u
            for q in range(Q):
                A_q = self.SCM_approximation.truth_problem.operator["a"][q]
                UB_vector[q] = (transpose(u)*A_q*u)/norm_S_squared
            return UB_vector
            
        def update_UB_vectors_J(self, UB_vector):
            self.SCM_approximation.UB_vectors_J.append(UB_vector)
            self.SCM_approximation.UB_vectors_J.save(self.SCM_approximation.folder["reduced_operators"], "UB_vectors_J")
            
        ## Choose the next parameter in the offline stage in a greedy fashion
        def greedy(self):
            def solve_and_estimate_error(mu, index):
                self.offline.__func__.mu_index = index
                self.SCM_approximation.set_mu(mu)
                
                LB = self.SCM_approximation.get_stability_factor_lower_bound(mu, False)
                UB = self.SCM_approximation.get_stability_factor_upper_bound(mu)
                error_estimator = (UB - LB)/UB
                
                from numpy import isclose
                if LB/UB < 0 and not isclose(LB/UB, 0.): # if LB/UB << 0
                    print("SCM warning at mu =", mu , ": LB =", LB, "< 0")
                if LB/UB > 1 and not isclose(LB/UB, 1.): # if LB/UB >> 1
                    print("SCM warning at mu =", mu , ": LB =", LB, "> UB =", UB)
                    
                self.SCM_approximation.alpha_LB_on_xi_train[index] = max(0, LB)
                return error_estimator
                
            (error_estimator_max, error_estimator_argmax) = self.xi_train.max(solve_and_estimate_error)
            print("maximum SCM error estimator =", error_estimator_max)
            self.SCM_approximation.set_mu(self.xi_train[error_estimator_argmax])
            self.offline.__func__.mu_index = error_estimator_argmax
            self.greedy_selected_parameters.append(self.xi_train[error_estimator_argmax])
            self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
            self.greedy_error_estimators.append(error_estimator_max)
            self.greedy_error_estimators.save(self.folder["post_processing"], "error_estimator_max")
            self.SCM_approximation.alpha_LB_on_xi_train.save(self.SCM_approximation.folder["reduced_operators"], "alpha_LB_on_xi_train")
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
        
        ## Initialize data structures required for the error analysis phase
        @override
        def _init_error_analysis(self):
            # Initialize the exact coercivity constant object
            self.SCM_approximation.exact_coercivity_constant_calculator.init()
            
            # Initialize reduced order data structures in the SCM online problem
            self.SCM_approximation.init("online")
        
        # Compute the error of the empirical interpolation approximation with respect to the
        # exact function over the test set
        @override
        def error_analysis(self, N=None):
            if N is None:
                N = self.SCM_approximation.N
                
            self._init_error_analysis()
            
            print("==============================================================")
            print("=             SCM error analysis begins                      =")
            print("==============================================================")
            print("")
            
            error_analysis_table = ErrorAnalysisTable(self.xi_test)
            error_analysis_table.set_Nmin(N)
            error_analysis_table.set_Nmax(N)
            error_analysis_table.add_column("normalized_error", group_name="scm", operations=("min", "mean", "max"))
            
            for run in range(len(self.xi_test)):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run =", run, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                
                self.SCM_approximation.set_mu(self.xi_test[run])
                
                (exact, _) = self.SCM_approximation.exact_coercivity_constant_calculator.solve()
                LB = self.SCM_approximation.get_stability_factor_lower_bound(self.SCM_approximation.mu, False)
                UB = self.SCM_approximation.get_stability_factor_upper_bound(self.SCM_approximation.mu)
                
                from numpy import isclose
                if LB/UB < 0 and not isclose(LB/UB, 0.): # if LB/UB << 0
                    print("SCM warning at mu =", self.SCM_approximation.mu , ": LB =", LB, "< 0")
                if LB/UB > 1 and not isclose(LB/UB, 1.): # if LB/UB >> 1
                    print("SCM warning at mu =", self.SCM_approximation.mu , ": LB =", LB, "> UB =", UB)
                if LB/exact > 1 and not isclose(LB/exact, 1.): # if LB/exact >> 1
                    print("SCM warning at mu =", self.SCM_approximation.mu , ": LB =", LB, "> exact =", exact)
                
                error_analysis_table["normalized_error", N, run] = (exact - LB)/UB
            
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
    
    @extends(ReductionMethod_DerivedClass, preserve_class_name=True)
    class SCMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(self, truth_problem)
            
            # Storage for SCM reduction method
            self.SCM_reduction = _SCMReductionMethod(self.truth_problem.SCM_approximation, type(self.truth_problem).__name__ + "/scm")
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the SCM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        @override
        def set_Nmax(self, Nmax, **kwargs):
            ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "SCM" in kwargs
            Nmax_SCM = kwargs["SCM"]
            assert isinstance(Nmax_SCM, int)
            self.SCM_reduction.set_Nmax(Nmax_SCM) # kwargs are not needed

            
        ## OFFLINE: set the elements in the training set \xi_train.
        @override
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_train(self, ntrain, enable_import, sampling)
            import_successful_SCM = self.SCM_reduction.set_xi_train(ntrain, enable_import=True, sampling=sampling)
            return import_successful and import_successful_SCM
            
        ## ERROR ANALYSIS: set the elements in the test set \xi_test.
        @override
        def set_xi_test(self, ntest, enable_import=False, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_test(self, ntest, enable_import, sampling)
            import_successful_SCM = self.SCM_reduction.set_xi_test(ntest, enable_import, sampling)
            return import_successful and import_successful_SCM
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
    
        ## Perform the offline phase of the reduced order model
        @override
        def offline(self):
            # Perform first the SCM offline phase, ...
            bak_first_mu = tuple(list(self.truth_problem.mu))
            self.SCM_reduction.offline()
            # ..., and then call the parent method.
            self.truth_problem.set_mu(bak_first_mu)
            return ReductionMethod_DerivedClass.offline(self)
    
        #  @}
        ########################### end - OFFLINE STAGE - end ###########################
    
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
    
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the test set
        @override
        def error_analysis(self, N=None):
            # Perform first the SCM error analysis, ...
            self.SCM_reduction.error_analysis(N)
            # ..., and then call the parent method.
            ReductionMethod_DerivedClass.error_analysis(self, N)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return SCMDecoratedReductionMethod_Class
    
