# Copyright (C) 2015-2018 by the RBniCS authors
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

import os
from numpy import isclose
from rbnics.backends import transpose
from rbnics.backends.online import OnlineVector
from rbnics.reduction_methods.base import ReductionMethod
from rbnics.utils.io import ErrorAnalysisTable, Folders, GreedyErrorEstimatorsList, SpeedupAnalysisTable, Timer
from rbnics.scm.problems import ParametrizedCoercivityConstantEigenProblem

# Empirical interpolation method for the interpolation of parametrized functions
class SCMApproximationReductionMethod(ReductionMethod):
    
    # Default initialization of members
    def __init__(self, SCM_approximation, folder_prefix):
        # Call the parent initialization
        ReductionMethod.__init__(self, folder_prefix)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.SCM_approximation = SCM_approximation
        # I/O
        self.folder["post_processing"] = os.path.join(self.folder_prefix, "post_processing")
        self.greedy_selected_parameters = SCM_approximation.greedy_selected_parameters
        self.greedy_error_estimators = GreedyErrorEstimatorsList()
        
        # Get data that were temporarily store in the SCM_approximation
        self.bounding_box_minimum_eigensolver_parameters = self.SCM_approximation._input_storage_for_SCM_reduction["bounding_box_minimum_eigensolver_parameters"]
        self.bounding_box_maximum_eigensolver_parameters = self.SCM_approximation._input_storage_for_SCM_reduction["bounding_box_maximum_eigensolver_parameters"]
        del self.SCM_approximation._input_storage_for_SCM_reduction

    # OFFLINE: set the elements in the training set.
    def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
        assert enable_import
        import_successful = ReductionMethod.initialize_training_set(self, self.SCM_approximation.mu_range, ntrain, enable_import, sampling, **kwargs)
        self.SCM_approximation.training_set = self.training_set
        return import_successful
        
    def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
        return ReductionMethod.initialize_testing_set(self, self.SCM_approximation.mu_range, ntest, enable_import, sampling, **kwargs)
        
    # Perform the offline phase of SCM
    def offline(self):
        need_to_do_offline_stage = self._init_offline()
        if need_to_do_offline_stage:
            self._offline()
        self._finalize_offline()
        return self.SCM_approximation
    
    # Initialize data structures required for the offline phase
    def _init_offline(self):
        # Prepare folders and init SCM approximation
        all_folders = Folders()
        all_folders.update(self.folder)
        all_folders.update(self.SCM_approximation.folder)
        all_folders.pop("testing_set") # this is required only in the error/speedup analysis
        all_folders.pop("error_analysis") # this is required only in the error analysis
        all_folders.pop("speedup_analysis") # this is required only in the speedup analysis
        at_least_one_folder_created = all_folders.create()
        if not at_least_one_folder_created:
            return False # offline construction should be skipped, since data are already available
        else:
            self.SCM_approximation.init("offline")
            return True # offline construction should be carried out
            
    def _offline(self):
        print("==============================================================")
        print("=" + "{:^60}".format("SCM offline phase begins") + "=")
        print("==============================================================")
        print("")
        
        # Compute the bounding box \mathcal{B}
        self.compute_bounding_box()
        print("")
        
        # Arbitrarily start from the first parameter in the training set
        self.SCM_approximation.set_mu(self.training_set[0])
        relative_error_estimator_max = 2.*self.tol
        
        while self.SCM_approximation.N < self.Nmax and relative_error_estimator_max >= self.tol:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM N =", self.SCM_approximation.N, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            # Store the greedy parameter
            self.store_greedy_selected_parameters()
            
            # Evaluate the coercivity constant
            print("evaluate the stability factor for mu =", self.SCM_approximation.mu)
            (alpha, eigenvector) = self.SCM_approximation.evaluate_stability_factor()
            print("stability factor =", alpha)
            
            # Update data structures related to upper bound vectors
            UB_vector = self.compute_UB_vector(eigenvector)
            self.update_UB_vectors(UB_vector)
            
            # Prepare for next iteration
            print("find next mu")
            (error_estimator_max, relative_error_estimator_max) = self.greedy()
            print("maximum SCM error estimator =", error_estimator_max)
            print("maximum SCM relative error estimator =", relative_error_estimator_max)
            
            print("")
        
        print("==============================================================")
        print("=" + "{:^60}".format("SCM offline phase ends") + "=")
        print("==============================================================")
        print("")
        
    # Finalize data structures required after the offline phase
    def _finalize_offline(self):
        self.SCM_approximation.init("online")
        
    # Compute the bounding box \mathcal{B}
    def compute_bounding_box(self):
        # Resize the bounding box storage
        Q = self.SCM_approximation.truth_problem.Q["a"]
        
        for q in range(Q):
            # Compute the minimum eigenvalue
            minimum_eigenvalue_calculator = ParametrizedCoercivityConstantEigenProblem(self.SCM_approximation.truth_problem, ("a", q), False, "smallest", self.bounding_box_minimum_eigensolver_parameters, self.folder_prefix)
            minimum_eigenvalue_calculator.init()
            (self.SCM_approximation.B_min[q], _) = minimum_eigenvalue_calculator.solve()
            print("B_min[" + str(q) + "] = " + str(self.SCM_approximation.B_min[q]))
            
            # Compute the maximum eigenvalue
            maximum_eigenvalue_calculator = ParametrizedCoercivityConstantEigenProblem(self.SCM_approximation.truth_problem, ("a", q), False, "largest", self.bounding_box_maximum_eigensolver_parameters, self.folder_prefix)
            maximum_eigenvalue_calculator.init()
            (self.SCM_approximation.B_max[q], _) = maximum_eigenvalue_calculator.solve()
            print("B_max[" + str(q) + "] = " + str(self.SCM_approximation.B_max[q]))
        
        # Save to file
        self.SCM_approximation.B_min.save(self.SCM_approximation.folder["reduced_operators"], "B_min")
        self.SCM_approximation.B_max.save(self.SCM_approximation.folder["reduced_operators"], "B_max")
        
    # Store the greedy parameter
    def store_greedy_selected_parameters(self):
        mu = self.SCM_approximation.mu
        
        self.SCM_approximation.greedy_selected_parameters.append(mu)
        self.SCM_approximation.N = len(self.SCM_approximation.greedy_selected_parameters)
        
        # Save to file
        self.SCM_approximation.greedy_selected_parameters.save(self.SCM_approximation.folder["reduced_operators"], "greedy_selected_parameters")
        
    # Compute the ratio between a_q(u,u) and s(u,u), for all q in vec
    def compute_UB_vector(self, u):
        Q = self.SCM_approximation.truth_problem.Q["a"]
        inner_product = self.SCM_approximation.truth_problem.inner_product[0]
        UB_vector = OnlineVector(Q)
        norm_S_squared = transpose(u)*inner_product*u
        for q in range(Q):
            A_q = self.SCM_approximation.truth_problem.operator["a"][q]
            UB_vector[q] = (transpose(u)*A_q*u)/norm_S_squared
        return UB_vector
        
    def update_UB_vectors(self, UB_vector):
        self.SCM_approximation.UB_vectors.append(UB_vector)
        self.SCM_approximation.UB_vectors.save(self.SCM_approximation.folder["reduced_operators"], "UB_vectors")
        
    # Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        def solve_and_estimate_error(mu):
            self.SCM_approximation.set_mu(mu)
            
            LB = self.SCM_approximation.get_stability_factor_lower_bound()
            UB = self.SCM_approximation.get_stability_factor_upper_bound()
            error_estimator = (UB - LB)/UB
            
            if LB/UB < 0 and not isclose(LB/UB, 0.): # if LB/UB << 0
                print("SCM warning at mu = " + str(mu) + ": LB = " + str(LB) + " < 0")
            if LB/UB > 1 and not isclose(LB/UB, 1.): # if LB/UB >> 1
                print("SCM warning at mu = " + str(mu) + ": LB = " + str(LB) + " > UB = " + str(UB))
                
            return error_estimator
            
        (error_estimator_max, error_estimator_argmax) = self.training_set.max(solve_and_estimate_error)
        self.SCM_approximation.set_mu(self.training_set[error_estimator_argmax])
        self.greedy_error_estimators.append(error_estimator_max)
        self.greedy_error_estimators.save(self.folder["post_processing"], "error_estimator_max")
        return (error_estimator_max, error_estimator_max/self.greedy_error_estimators[0])
        
    # Initialize data structures required for the error analysis phase
    def _init_error_analysis(self, **kwargs):
        # Initialize the exact coercivity constant object
        self.SCM_approximation.exact_coercivity_constant_calculator.init()
        
        # Initialize reduced order data structures in the SCM online problem
        self.SCM_approximation.init("online")
    
    # Compute the error of the scm approximation with respect to the
    # exact coercivity over the testing set
    def error_analysis(self, N=None, filename=None, **kwargs):
        if N is None:
            N = self.SCM_approximation.N
        assert len(kwargs) == 0 # not used in this method
        
        self._init_error_analysis(**kwargs)
        self._error_analysis(N, filename, **kwargs)
        self._finalize_error_analysis(**kwargs)
        
    def _error_analysis(self, N=None, filename=None, **kwargs):
        print("==============================================================")
        print("=" + "{:^60}".format("SCM error analysis begins") + "=")
        print("==============================================================")
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.testing_set)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("normalized_error", group_name="scm", operations=("min", "mean", "max"))
        
        for (run, mu) in enumerate(self.testing_set):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run =", run, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            self.SCM_approximation.set_mu(mu)
            
            (exact, _) = self.SCM_approximation.evaluate_stability_factor()
            for n in range(1, N + 1): # n = 1, ... N
                LB = self.SCM_approximation.get_stability_factor_lower_bound(n)
                UB = self.SCM_approximation.get_stability_factor_upper_bound(n)
                
                if LB/UB < 0 and not isclose(LB/UB, 0.): # if LB/UB << 0
                    print("SCM warning at mu = " + str(mu) + ": LB = " + str(LB) + " < 0")
                if LB/UB > 1 and not isclose(LB/UB, 1.): # if LB/UB >> 1
                    print("SCM warning at mu = " + str(mu) + ": LB = " + str(LB) + " > UB = " + str(UB))
                if LB/exact > 1 and not isclose(LB/exact, 1.): # if LB/exact >> 1
                    print("SCM warning at mu = " + str(mu) + ": LB = " + str(LB) + " > exact =" + str(exact))
                
                error_analysis_table["normalized_error", n, run] = (exact - LB)/UB
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print("==============================================================")
        print("=" + "{:^60}".format("SCM error analysis ends") + "=")
        print("==============================================================")
        print("")
        
        # Export error analysis table
        error_analysis_table.save(self.folder["error_analysis"], "error_analysis" if filename is None else filename)
        
    # Compute the speedup of the scm approximation with respect to the
    # exact coercivity over the testing set
    def speedup_analysis(self, N=None, filename=None, **kwargs):
        if N is None:
            N = self.SCM_approximation.N
        assert len(kwargs) == 0 # not used in this method
            
        self._init_speedup_analysis(**kwargs)
        self._speedup_analysis(N, filename, **kwargs)
        self._finalize_speedup_analysis(**kwargs)
        
    # Initialize data structures required for the speedup analysis phase
    def _init_speedup_analysis(self, **kwargs):
        # Make sure to clean up snapshot cache to ensure that parametrized
        # expression evaluation is actually carried out
        self.SCM_approximation._alpha_LB_cache.clear()
        self.SCM_approximation._alpha_UB_cache.clear()
        self.SCM_approximation.exact_coercivity_constant_calculator._eigenvalue_cache.clear()
        self.SCM_approximation.exact_coercivity_constant_calculator._eigenvector_cache.clear()
        
    def _speedup_analysis(self, N=None, filename=None, **kwargs):
        print("==============================================================")
        print("=" + "{:^60}".format("SCM speedup analysis begins") + "=")
        print("==============================================================")
        print("")
        
        speedup_analysis_table = SpeedupAnalysisTable(self.testing_set)
        speedup_analysis_table.set_Nmax(N)
        speedup_analysis_table.add_column("speedup", group_name="speedup", operations=("min", "mean", "max"))
        
        exact_timer = Timer("parallel")
        SCM_timer = Timer("serial")
        
        for (run, mu) in enumerate(self.testing_set):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run =", run, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            self.SCM_approximation.set_mu(mu)
            
            exact_timer.start()
            self.SCM_approximation.evaluate_stability_factor()
            elapsed_exact = exact_timer.stop()
            
            for n in range(1, N + 1): # n = 1, ... N
                SCM_timer.start()
                self.SCM_approximation.get_stability_factor_lower_bound(n)
                self.SCM_approximation.get_stability_factor_upper_bound(n)
                elapsed_SCM = SCM_timer.stop()
                speedup_analysis_table["speedup", n, run] = elapsed_exact/elapsed_SCM
        
        # Print
        print("")
        print(speedup_analysis_table)
        
        print("")
        print("==============================================================")
        print("=" + "{:^60}".format("SCM speedup analysis ends") + "=")
        print("==============================================================")
        print("")
        
        # Export speedup analysis table
        speedup_analysis_table.save(self.folder["speedup_analysis"], "speedup_analysis" if filename is None else filename)
