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

import types
from rbnics.backends.common.linear_program_solver import LinearProgramSolver
from rbnics.utils.decorators import Extends, override, ReductionMethodDecoratorFor
from rbnics.scm.problems import SCM
from rbnics.scm.reduction_methods.scm_approximation_reduction_method import SCMApproximationReductionMethod

@ReductionMethodDecoratorFor(SCM)
def SCMDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class SCMDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Storage for SCM reduction method
            self.SCM_reduction = SCMApproximationReductionMethod(self.truth_problem.SCM_approximation, type(self.truth_problem).__name__ + "/scm")
            
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        @override
        def set_Nmax(self, Nmax, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "SCM" in kwargs
            Nmax_SCM = kwargs["SCM"]
            assert isinstance(Nmax_SCM, int)
            self.SCM_reduction.set_Nmax(Nmax_SCM) # kwargs are not needed

        ## OFFLINE: set tolerance (stopping criterion)
        @override
        def set_tolerance(self, tol, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
            assert "SCM" in kwargs
            tol_SCM = kwargs["SCM"]
            assert isinstance(tol_SCM, float)
            self.SCM_reduction.set_tolerance(tol_SCM) # kwargs are not needed
            
        ## OFFLINE: set the elements in the training set.
        @override
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
            # Initialize training set of SCM reduction
            assert "SCM" in kwargs
            ntrain_SCM = kwargs["SCM"]
            import_successful_SCM = self.SCM_reduction.initialize_training_set(ntrain_SCM, enable_import=True, sampling=sampling) # kwargs are not needed
            if LinearProgramSolver.solve_can_raise():
                # In an exception is raised we will fall back to exact evaluation is required, 
                # and thus we cannot use a distributed training set
                self.training_set.distributed_max = False
                self.SCM_reduction.training_set.distributed_max = False
            # Return
            return import_successful and import_successful_SCM
            
        ## ERROR ANALYSIS: set the elements in the testing set.
        @override
        def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
            # Initialize testing set of SCM reduction
            assert "SCM" in kwargs
            ntest_SCM = kwargs["SCM"]
            import_successful_SCM = self.SCM_reduction.initialize_testing_set(ntest_SCM, enable_import, sampling) # kwargs are not needed
            # Return
            return import_successful and import_successful_SCM
            
        ## Perform the offline phase of the reduced order model
        @override
        def offline(self):
            # Perform first the SCM offline phase, ...
            bak_first_mu = self.truth_problem.mu
            self.SCM_reduction.offline()
            # ..., and then call the parent method.
            self.truth_problem.set_mu(bak_first_mu)
            return DifferentialProblemReductionMethod_DerivedClass.offline(self)
            
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def error_analysis(self, N=None, **kwargs):
            # Perform first the SCM error analysis, ...
            if (
                "with_respect_to" not in kwargs # otherwise we assume the user was interested in computing the error w.r.t. 
                                                # an exact coercivity constant, 
                                                # so he probably is not interested in the error analysis of SCM
                    and
                "SCM" not in kwargs             # otherwise we assume the user was interested in computing the error for a fixed number of SCM basis
                                                # functions, thus he has already carried out the error analysis of SCM
            ):
                self.SCM_reduction.error_analysis(N)
            # ..., and then call the parent method.
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N, **kwargs)
            
        @override
        def _init_error_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_error_analysis(self, **kwargs)
            # Replace stability factor computation, if needed
            self._replace_stability_factor_computation(**kwargs)
            
        @override
        def _finalize_error_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_error_analysis(self, **kwargs)
            # Undo replacement of stability factor computation, if needed
            self._undo_replace_stability_factor_computation(**kwargs)
            
        # Compute the speedup of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def speedup_analysis(self, N=None, **kwargs):
            # Perform first the SCM speedup analysis, ...
            if (
                "with_respect_to" not in kwargs # otherwise we assume the user was interested in computing the speedup w.r.t. 
                                                # an exact coercivity constant, 
                                                # so he probably is not interested in the speedup analysis of SCM
                    and
                "SCM" not in kwargs             # otherwise we assume the user was interested in computing the speedup for a fixed number of SCM basis
                                                # functions, thus he has already carried out the speedup analysis of SCM
            ):
                self.SCM_reduction.speedup_analysis(N)
            # ..., and then call the parent method.
            DifferentialProblemReductionMethod_DerivedClass.speedup_analysis(self, N, **kwargs)
            
        @override
        def _init_speedup_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_speedup_analysis(self, **kwargs)
            # Replace stability factor computation, if needed
            self._replace_stability_factor_computation(**kwargs)
            
        @override
        def _finalize_speedup_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_speedup_analysis(self, **kwargs)
            # Undo replacement of stability factor computation, if needed
            self._undo_replace_stability_factor_computation(**kwargs)
            
        def _replace_stability_factor_computation(self, **kwargs):
            self._replace_stability_factor_computation__get_stability_factor__original = self.reduced_problem.get_stability_factor
            if "SCM" not in kwargs:
                if "with_respect_to" in kwargs:
                    # Assume that the user wants to disable SCM and use the exact stability factor
                    def replaced_get_stability_factor(self_):
                        return kwargs["with_respect_to"].get_stability_factor()
                    self.reduced_problem.get_stability_factor = types.MethodType(replaced_get_stability_factor, self.reduced_problem)
            else:
                assert isinstance(kwargs["SCM"], int)
                # Assume that the user wants to use SCM with a prescribed number of basis functions
                def replaced_get_stability_factor(self_):
                    return self.truth_problem.SCM_approximation.get_stability_factor_lower_bound(kwargs["SCM"])
                self.reduced_problem.get_stability_factor = types.MethodType(replaced_get_stability_factor, self.reduced_problem)
            
        def _undo_replace_stability_factor_computation(self, **kwargs):
            self.reduced_problem.get_stability_factor = self._replace_stability_factor_computation__get_stability_factor__original
            del self._replace_stability_factor_computation__get_stability_factor__original
        
    # return value (a class) for the decorator
    return SCMDecoratedReductionMethod_Class
    
