# Copyright (C) 2015-2019 by the RBniCS authors
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
from rbnics.scm.problems import ParametrizedStabilityFactorEigenProblem
from rbnics.utils.io import ErrorAnalysisTable, Folders, GreedyErrorEstimatorsList, SpeedupAnalysisTable, TextBox, TextLine, Timer

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
        required_folders = Folders()
        required_folders.update(self.folder)
        required_folders.update(self.SCM_approximation.folder)
        optional_folders = Folders()
        optional_folders["cache"] = required_folders.pop("cache") # this does not affect the availability of offline data
        optional_folders["testing_set"] = required_folders.pop("testing_set") # this is required only in the error/speedup analysis
        optional_folders["error_analysis"] = required_folders.pop("error_analysis") # this is required only in the error analysis
        optional_folders["speedup_analysis"] = required_folders.pop("speedup_analysis") # this is required only in the speedup analysis
        at_least_one_required_folder_created = required_folders.create()
        at_least_one_optional_folder_created = optional_folders.create()  # noqa: F841
        if not at_least_one_required_folder_created:
            return False # offline construction should be skipped, since data are already available
        else:
            self.SCM_approximation.init("offline")
            return True # offline construction should be carried out
            
    def _offline(self):
        print(TextBox("SCM offline phase begins", fill="="))
        print("")
        
        # Compute the bounding box \mathcal{B}
        self.compute_bounding_box()
        print("")
        
        # Arbitrarily start from the first parameter in the training set
        self.SCM_approximation.set_mu(self.training_set[0])
        relative_error_estimator_max = 2.*self.tol
        
        while self.SCM_approximation.N < self.Nmax and relative_error_estimator_max >= self.tol:
            print(TextLine("SCM N = " + str(self.SCM_approximation.N), fill="~"))
            
            # Store the greedy parameter
            self.store_greedy_selected_parameters()
            
            # Evaluate the stability factor
            print("evaluate the stability factor for mu =", self.SCM_approximation.mu)
            (stability_factor, eigenvector) = self.SCM_approximation.evaluate_stability_factor()
            print("stability factor =", stability_factor)
            
            # Update data structures related to upper bound vectors
            upper_bound_vector = self.compute_upper_bound_vector(eigenvector)
            self.update_upper_bound_vectors(upper_bound_vector)
            
            # Prepare for next iteration
            print("find next mu")
            (error_estimator_max, relative_error_estimator_max) = self.greedy()
            print("maximum SCM error estimator =", error_estimator_max)
            print("maximum SCM relative error estimator =", relative_error_estimator_max)
            
            print("")
            
        print(TextBox("SCM offline phase ends", fill="="))
        print("")
        
    # Finalize data structures required after the offline phase
    def _finalize_offline(self):
        self.SCM_approximation.init("online")
        
    # Compute the bounding box \mathcal{B}
    def compute_bounding_box(self):
        # Resize the bounding box storage
        Q = self.SCM_approximation.truth_problem.Q["stability_factor_left_hand_matrix"]
        
        for q in range(Q):
            # Compute the minimum eigenvalue
            minimum_eigenvalue_calculator = ParametrizedStabilityFactorEigenProblem(self.SCM_approximation.truth_problem, "smallest", self.SCM_approximation.truth_problem._eigen_solver_parameters["bounding_box_minimum"], self.folder_prefix, expansion_index=q)
            minimum_eigenvalue_calculator.init()
            (self.SCM_approximation.bounding_box_min[q], _) = minimum_eigenvalue_calculator.solve()
            print("bounding_box_min[" + str(q) + "] = " + str(self.SCM_approximation.bounding_box_min[q]))
            
            # Compute the maximum eigenvalue
            maximum_eigenvalue_calculator = ParametrizedStabilityFactorEigenProblem(self.SCM_approximation.truth_problem, "largest", self.SCM_approximation.truth_problem._eigen_solver_parameters["bounding_box_maximum"], self.folder_prefix, expansion_index=q)
            maximum_eigenvalue_calculator.init()
            (self.SCM_approximation.bounding_box_max[q], _) = maximum_eigenvalue_calculator.solve()
            print("bounding_box_max[" + str(q) + "] = " + str(self.SCM_approximation.bounding_box_max[q]))
        
        # Save to file
        self.SCM_approximation.bounding_box_min.save(self.SCM_approximation.folder["reduced_operators"], "bounding_box_min")
        self.SCM_approximation.bounding_box_max.save(self.SCM_approximation.folder["reduced_operators"], "bounding_box_max")
        
    # Store the greedy parameter
    def store_greedy_selected_parameters(self):
        mu = self.SCM_approximation.mu
        
        self.SCM_approximation.greedy_selected_parameters.append(mu)
        self.SCM_approximation.N = len(self.SCM_approximation.greedy_selected_parameters)
        
        # Save to file
        self.SCM_approximation.greedy_selected_parameters.save(self.SCM_approximation.folder["reduced_operators"], "greedy_selected_parameters")
        
    def compute_upper_bound_vector(self, u):
        Q = self.SCM_approximation.truth_problem.Q["stability_factor_left_hand_matrix"]
        A = self.SCM_approximation.truth_problem.operator["stability_factor_left_hand_matrix"]
        B = self.SCM_approximation.truth_problem.operator["stability_factor_right_hand_matrix"]
        assert len(B) == 1
        normalization = transpose(u)*B[0]*u
        upper_bound_vector = OnlineVector(Q)
        for q in range(Q):
            upper_bound_vector[q] = (transpose(u)*A[q]*u)/normalization
        return upper_bound_vector
        
    def update_upper_bound_vectors(self, upper_bound_vector):
        self.SCM_approximation.upper_bound_vectors.append(upper_bound_vector)
        self.SCM_approximation.upper_bound_vectors.save(self.SCM_approximation.folder["reduced_operators"], "upper_bound_vectors")
        
    # Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        def solve_and_estimate_error(mu):
            self.SCM_approximation.set_mu(mu)
            
            stability_factor_lower_bound = self.SCM_approximation.get_stability_factor_lower_bound()
            stability_factor_upper_bound = self.SCM_approximation.get_stability_factor_upper_bound()
            ratio = stability_factor_lower_bound/stability_factor_upper_bound
            
            if ratio < 0. and not isclose(ratio, 0.): # if ratio << 0
                print("SCM warning at mu = " + str(mu) + ": stability factor lower bound = " + str(stability_factor_lower_bound) + " < 0")
            if ratio > 1. and not isclose(ratio, 1.): # if ratio >> 1
                print("SCM warning at mu = " + str(mu) + ": stability factor lower bound = " + str(stability_factor_lower_bound) + " > stability factor upper bound = " + str(stability_factor_upper_bound))
                
            error_estimator = 1. - ratio
            return error_estimator
            
        (error_estimator_max, error_estimator_argmax) = self.training_set.max(solve_and_estimate_error)
        self.SCM_approximation.set_mu(self.training_set[error_estimator_argmax])
        self.greedy_error_estimators.append(error_estimator_max)
        self.greedy_error_estimators.save(self.folder["post_processing"], "error_estimator_max")
        return (error_estimator_max, error_estimator_max/self.greedy_error_estimators[0])
        
    # Initialize data structures required for the error analysis phase
    def _init_error_analysis(self, **kwargs):
        # Initialize reduced order data structures in the SCM online problem
        self.SCM_approximation.init("online")
    
    # Compute the error of the scm approximation with respect to the
    # exact stability factor over the testing set
    def error_analysis(self, N_generator=None, filename=None, **kwargs):
        assert len(kwargs) == 0 # not used in this method
        
        self._init_error_analysis(**kwargs)
        self._error_analysis(N_generator, filename, **kwargs)
        self._finalize_error_analysis(**kwargs)
        
    def _error_analysis(self, N_generator=None, filename=None, **kwargs):
        if N_generator is None:
            def N_generator(n):
                return n
                
        N = self.SCM_approximation.N
        
        print(TextBox("SCM error analysis begins", fill="="))
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.testing_set)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("normalized_error", group_name="scm", operations=("min", "mean", "max"))
        
        for (mu_index, mu) in enumerate(self.testing_set):
            print(TextLine("SCM " + str(mu_index), fill="~"))
            
            self.SCM_approximation.set_mu(mu)
            
            (exact_stability_factor, _) = self.SCM_approximation.evaluate_stability_factor()
            for n in range(1, N + 1): # n = 1, ... N
                n_arg = N_generator(n)
                
                if n_arg is not None:
                    stability_factor_lower_bound = self.SCM_approximation.get_stability_factor_lower_bound(n_arg)
                    stability_factor_upper_bound = self.SCM_approximation.get_stability_factor_upper_bound(n_arg)
                    ratio_lower_bound_to_upper_bound = stability_factor_lower_bound/stability_factor_upper_bound
                    ratio_lower_bound_to_exact = stability_factor_lower_bound/exact_stability_factor
                    
                    if ratio_lower_bound_to_upper_bound < 0. and not isclose(ratio_lower_bound_to_upper_bound, 0.): # if ratio_lower_bound_to_upper_bound << 0
                        print("SCM warning at mu = " + str(mu) + ": stability factor lower bound = " + str(stability_factor_lower_bound) + " < 0")
                    if ratio_lower_bound_to_upper_bound > 1. and not isclose(ratio_lower_bound_to_upper_bound, 1.): # if ratio_lower_bound_to_upper_bound >> 1
                        print("SCM warning at mu = " + str(mu) + ": stability factor lower bound = " + str(stability_factor_lower_bound) + " > stability factor upper bound = " + str(stability_factor_upper_bound))
                    if ratio_lower_bound_to_exact > 1. and not isclose(ratio_lower_bound_to_exact, 1.): # if ratio_lower_bound_to_exact >> 1
                        print("SCM warning at mu = " + str(mu) + ": stability factor lower bound = " + str(stability_factor_lower_bound) + " > exact stability factor =" + str(exact_stability_factor))
                    
                    error_analysis_table["normalized_error", n, mu_index] = (exact_stability_factor - stability_factor_lower_bound)/stability_factor_upper_bound
                else:
                    error_analysis_table["normalized_error", n, mu_index] = NotImplemented
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print(TextBox("SCM error analysis ends", fill="="))
        print("")
        
        # Export error analysis table
        error_analysis_table.save(self.folder["error_analysis"], "error_analysis" if filename is None else filename)
        
    # Compute the speedup of the scm approximation with respect to the
    # exact stability factor over the testing set
    def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
        assert len(kwargs) == 0 # not used in this method
            
        self._init_speedup_analysis(**kwargs)
        self._speedup_analysis(N_generator, filename, **kwargs)
        self._finalize_speedup_analysis(**kwargs)
        
    # Initialize data structures required for the speedup analysis phase
    def _init_speedup_analysis(self, **kwargs):
        # Make sure to clean up snapshot cache to ensure that parametrized
        # expression evaluation is actually carried out
        self.SCM_approximation._stability_factor_lower_bound_cache.clear()
        self.SCM_approximation._stability_factor_upper_bound_cache.clear()
        self.SCM_approximation.stability_factor_calculator._eigenvalue_cache.clear()
        self.SCM_approximation.stability_factor_calculator._eigenvector_cache.clear()
        
    def _speedup_analysis(self, N_generator=None, filename=None, **kwargs):
        if N_generator is None:
            def N_generator(n):
                return n
                
        N = self.SCM_approximation.N
                
        print(TextBox("SCM speedup analysis begins", fill="="))
        print("")
        
        speedup_analysis_table = SpeedupAnalysisTable(self.testing_set)
        speedup_analysis_table.set_Nmax(N)
        speedup_analysis_table.add_column("speedup", group_name="speedup", operations=("min", "mean", "max"))
        
        exact_timer = Timer("parallel")
        SCM_timer = Timer("serial")
        
        for (mu_index, mu) in enumerate(self.testing_set):
            print(TextLine("SCM " + str(mu_index), fill="~"))
            
            self.SCM_approximation.set_mu(mu)
            
            exact_timer.start()
            self.SCM_approximation.evaluate_stability_factor()
            elapsed_exact = exact_timer.stop()
            
            for n in range(1, N + 1): # n = 1, ... N
                n_arg = N_generator(n)
                
                if n_arg is not None:
                    SCM_timer.start()
                    self.SCM_approximation.get_stability_factor_lower_bound(n_arg)
                    self.SCM_approximation.get_stability_factor_upper_bound(n_arg)
                    elapsed_SCM = SCM_timer.stop()
                    speedup_analysis_table["speedup", n, mu_index] = elapsed_exact/elapsed_SCM
                else:
                    speedup_analysis_table["speedup", n, mu_index] = NotImplemented
        
        # Print
        print("")
        print(speedup_analysis_table)
        
        print("")
        print(TextBox("SCM speedup analysis ends", fill="="))
        print("")
        
        # Export speedup analysis table
        speedup_analysis_table.save(self.folder["speedup_analysis"], "speedup_analysis" if filename is None else filename)
