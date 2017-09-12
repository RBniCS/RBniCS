# Copyright (C) 2015-2017 by the RBniCS authors
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
from rbnics.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable
from rbnics.utils.decorators import PreserveClassName

def PrimalDualReductionMethod(DualProblem):
    def PrimalDualReductionMethod_Decorator(DifferentialProblemReductionMethod_DerivedClass):
                
        @PreserveClassName
        class PrimalDualReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
    
            ## Default initialization of members
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
            
            ## OFFLINE: set maximum reduced space dimension (stopping criterion)
            def set_Nmax(self, Nmax, **kwargs):
                DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
                # Set Nmax of dual reduction
                assert "dual" in kwargs
                self.dual_reduction_method.set_Nmax(kwargs["dual"], **kwargs)
                
            ## OFFLINE: set tolerance (stopping criterion)
            def set_tolerance(self, tol, **kwargs):
                DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
                # Set tolerance of dual reduction
                assert "dual" in kwargs
                self.dual_reduction_method.set_tolerance(kwargs["dual"], **kwargs)
                
            ## OFFLINE: set the elements in the training set.
            def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
                import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
                # Initialize training set of dual reduction
                assert "dual" in kwargs
                import_successful_dual = self.dual_reduction_method.initialize_training_set(kwargs["dual"], enable_import, sampling, **kwargs)
                return import_successful and import_successful_dual
                
            ## ERROR ANALYSIS: set the elements in the testing set.
            def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
                import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
                # Initialize testing set of dual reduction
                assert "dual" in kwargs
                import_successful_dual = self.dual_reduction_method.initialize_testing_set(kwargs["dual"], enable_import, sampling, **kwargs)
                return import_successful and import_successful_dual
                
            ## Perform the offline phase of the reduced order model
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
            def error_analysis(self, N=None, **kwargs):
                # Carry out primal error analysis ...
                DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N, **kwargs)
                # ... and then dual error analysis
                ErrorAnalysisTable.suppress_group("output_error")
                ErrorAnalysisTable.suppress_group("output_relative_error")
                self._replace_dual_truth_problem(**kwargs)
                self.dual_reduction_method.error_analysis(N, **kwargs)
                self._undo_replace_dual_truth_problem(**kwargs)
                ErrorAnalysisTable.clear_suppressed_groups()
                
            # Compute the speedup of the reduced order approximation with respect to the full order one
            # over the testing set
            def speedup_analysis(self, N=None, **kwargs):
                # Carry out primal speedup analysis ...
                DifferentialProblemReductionMethod_DerivedClass.speedup_analysis(self, N, **kwargs)
                # ... and then dual speedup analysis
                SpeedupAnalysisTable.suppress_group("output_error")
                SpeedupAnalysisTable.suppress_group("output_relative_error")
                self._replace_dual_truth_problem(**kwargs)
                self.dual_reduction_method.speedup_analysis(N, **kwargs)
                self._undo_replace_dual_truth_problem(**kwargs)
                SpeedupAnalysisTable.clear_suppressed_groups()
                
            def _replace_dual_truth_problem(self, **kwargs):
                if "with_respect_to" in kwargs:
                    if not hasattr(self, "_replace_dual_truth_problem__bak_dual_truth_problem"):
                        self._replace_dual_truth_problem__bak_dual_truth_problem = self.dual_truth_problem
                        assert inspect.isfunction(kwargs["with_respect_to"])
                        self.dual_truth_problem = kwargs["with_respect_to"](self.dual_truth_problem)
                        self.dual_reduced_problem.truth_problem = self.dual_truth_problem
                
            def _undo_replace_dual_truth_problem(self, **kwargs):
                if "with_respect_to" in kwargs:
                    if hasattr(self, "_replace_dual_truth_problem__bak_dual_truth_problem"):
                        self.dual_truth_problem = self._replace_dual_truth_problem__bak_dual_truth_problem
                        self.dual_reduced_problem.truth_problem = self.dual_truth_problem
                        del self._replace_dual_truth_problem__bak_dual_truth_problem
                        
        # return value (a class) for the decorator
        return PrimalDualReductionMethod_Class
        
    # return the decorator itself
    return PrimalDualReductionMethod_Decorator
    
