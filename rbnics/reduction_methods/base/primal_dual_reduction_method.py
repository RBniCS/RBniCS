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

import inspect
from rbnics.backends import assign
from rbnics.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable
from rbnics.utils.decorators import PreserveClassName
from rbnics.utils.test import PatchInstanceMethod

def PrimalDualReductionMethod(DualProblem):
    def PrimalDualReductionMethod_Decorator(DifferentialProblemReductionMethod_DerivedClass):
                
        @PreserveClassName
        class PrimalDualReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
    
            # Default initialization of members
            def __init__(self, truth_problem, **kwargs):
                # Call the parent initialization
                DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
                
                # Dual high fidelity problem
                self.dual_truth_problem = DualProblem(truth_problem)
                # Dual reduction method
                self.dual_reduction_method = DifferentialProblemReductionMethod_DerivedClass(self.dual_truth_problem, **kwargs)
                # Dual reduced problem
                self.dual_reduced_problem = None
                
                # Change the folder names in dual reduction method
                new_folder_prefix = self.dual_truth_problem.folder_prefix
                for (key, name) in self.dual_reduction_method.folder.items():
                    self.dual_reduction_method.folder[key] = name.replace(self.dual_reduction_method.folder_prefix, new_folder_prefix)
                self.dual_reduction_method.folder_prefix = new_folder_prefix
                
                # Change the label in dual reduction method
                self.dual_reduction_method.label = "Dual " + self.dual_reduction_method.label
                
            # Propagate the values of all setters also to the dual object
            
            # OFFLINE: set maximum reduced space dimension (stopping criterion)
            def set_Nmax(self, Nmax, **kwargs):
                DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
                # Set Nmax of dual reduction
                assert "dual" in kwargs
                self.dual_reduction_method.set_Nmax(kwargs["dual"], **kwargs)
                
            # OFFLINE: set tolerance (stopping criterion)
            def set_tolerance(self, tol, **kwargs):
                DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
                # Set tolerance of dual reduction
                assert "dual" in kwargs
                self.dual_reduction_method.set_tolerance(kwargs["dual"], **kwargs)
                
            # OFFLINE: set the elements in the training set.
            def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
                import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
                # Initialize training set of dual reduction
                assert "dual" in kwargs
                import_successful_dual = self.dual_reduction_method.initialize_training_set(kwargs["dual"], enable_import, sampling, **kwargs)
                return import_successful and import_successful_dual
                
            # ERROR ANALYSIS: set the elements in the testing set.
            def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
                import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
                # Initialize testing set of dual reduction
                assert "dual" in kwargs
                import_successful_dual = self.dual_reduction_method.initialize_testing_set(kwargs["dual"], enable_import, sampling, **kwargs)
                return import_successful and import_successful_dual
                
            # Perform the offline phase of the reduced order model
            def offline(self):
                # Carry out primal offline stage ...
                bak_first_mu = self.truth_problem.mu
                primal_reduced_problem = DifferentialProblemReductionMethod_DerivedClass.offline(self)
                # ... and then dual offline stage
                self.truth_problem.set_mu(bak_first_mu)
                self.dual_reduced_problem = self.dual_reduction_method.offline()
                # Attach reduced dual problem to reduced primal problem, and viceversa
                primal_reduced_problem.dual_reduced_problem = self.dual_reduced_problem
                self.dual_reduced_problem.primal_reduced_problem = primal_reduced_problem
                # Compute data structures related to output correction and error estimation
                self.dual_reduced_problem.build_output_correction_and_estimation_operators()
                #
                return primal_reduced_problem
                
            # Compute the error of the reduced order approximation with respect to the full order one
            # over the testing set
            def error_analysis(self, N_generator=None, filename=None, **kwargs):
                # Carry out primal error analysis ...
                DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N_generator, filename, **kwargs)
                # ... and then dual error analysis
                ErrorAnalysisTable.suppress_group("output_error")
                ErrorAnalysisTable.suppress_group("output_relative_error")
                self.dual_reduction_method.error_analysis(N_generator, filename, **kwargs)
                ErrorAnalysisTable.clear_suppressed_groups()
                
            # Compute the speedup of the reduced order approximation with respect to the full order one
            # over the testing set
            def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
                # Carry out primal speedup analysis ...
                DifferentialProblemReductionMethod_DerivedClass.speedup_analysis(self, N_generator, filename, **kwargs)
                # ... and then dual speedup analysis
                SpeedupAnalysisTable.suppress_group("speedup_output")
                SpeedupAnalysisTable.suppress_group("speedup_output_and_estimate_error_output")
                SpeedupAnalysisTable.suppress_group("speedup_output_and_estimate_relative_error_output")
                self.dual_reduction_method.speedup_analysis(N_generator, filename, **kwargs)
                SpeedupAnalysisTable.clear_suppressed_groups()
                
            def _patch_truth_solve(self, force, **kwargs):
                DifferentialProblemReductionMethod_DerivedClass._patch_truth_solve(self, **kwargs)
                if "with_respect_to" in kwargs:
                    assert inspect.isfunction(kwargs["with_respect_to"])
                    other_dual_truth_problem = kwargs["with_respect_to"](self.dual_truth_problem)
                    def patched_dual_truth_solve(self_, **kwargs_):
                        other_dual_truth_problem.solve(**kwargs_)
                        assign(self.dual_truth_problem._solution, other_dual_truth_problem._solution)
                        return self.dual_truth_problem._solution
                        
                    self.patch_dual_truth_solve = PatchInstanceMethod(
                        self.dual_truth_problem,
                        "solve",
                        patched_dual_truth_solve
                    )
                    self.patch_dual_truth_solve.patch()
                    
                    # Initialize the affine expansion in the other dual truth problem
                    other_dual_truth_problem.init()
                else:
                    other_dual_truth_problem = self.dual_truth_problem
                    
                # Clean up solution caching and disable I/O
                if force:
                    # Make sure to clean up problem and reduced problem solution cache to ensure that
                    # solution and reduced solution are actually computed
                    other_dual_truth_problem._solution_cache.clear()
                    other_dual_truth_problem._output_cache.clear()
                    
                    # Disable the capability of importing/exporting dual truth solutions
                    def disable_import_solution_method(self_, folder=None, filename=None, solution=None, component=None, suffix=None):
                        raise OSError
                    self.disable_import_dual_solution = PatchInstanceMethod(other_dual_truth_problem, "import_solution", disable_import_solution_method)
                    def disable_export_solution_method(self_, folder=None, filename=None, solution=None, component=None, suffix=None):
                        pass
                    self.disable_export_dual_solution = PatchInstanceMethod(other_dual_truth_problem, "export_solution", disable_export_solution_method)
                    self.disable_import_dual_solution.patch()
                    self.disable_export_dual_solution.patch()
                
            def _undo_patch_truth_solve(self, force, **kwargs):
                DifferentialProblemReductionMethod_DerivedClass._undo_patch_truth_solve(self, **kwargs)
                if "with_respect_to" in kwargs:
                    self.patch_dual_truth_solve.unpatch()
                    del self.patch_dual_truth_solve
                    
                # Restore solution I/O
                if force:
                    # Restore the capability to import/export dual truth solutions
                    self.disable_import_dual_solution.unpatch()
                    self.disable_export_dual_solution.unpatch()
                    del self.disable_import_dual_solution
                    del self.disable_export_dual_solution
                    
        # return value (a class) for the decorator
        return PrimalDualReductionMethod_Class
        
    # return the decorator itself
    return PrimalDualReductionMethod_Decorator
