# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
import os
import inspect
from rbnics.scm.problems import SCM
from rbnics.scm.reduction_methods.scm_approximation_reduction_method import SCMApproximationReductionMethod
from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.utils.test import PatchInstanceMethod


@ReductionMethodDecoratorFor(SCM)
def SCMDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class SCMDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Storage for SCM reduction method
            self.SCM_reduction = SCMApproximationReductionMethod(
                self.truth_problem.SCM_approximation, os.path.join(self.truth_problem.name(), "scm"))

        # OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_Nmax(self, Nmax, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "SCM" in kwargs
            Nmax_SCM = kwargs["SCM"]
            assert isinstance(Nmax_SCM, int)
            self.SCM_reduction.set_Nmax(Nmax_SCM)  # kwargs are not needed

        # OFFLINE: set tolerance (stopping criterion)
        def set_tolerance(self, tol, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
            assert "SCM" in kwargs
            tol_SCM = kwargs["SCM"]
            assert isinstance(tol_SCM, Number)
            self.SCM_reduction.set_tolerance(tol_SCM)  # kwargs are not needed

        # OFFLINE: set the elements in the training set.
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(
                self, ntrain, enable_import, sampling, **kwargs)
            # Initialize training set of SCM reduction
            assert "SCM" in kwargs
            ntrain_SCM = kwargs["SCM"]
            import_successful_SCM = self.SCM_reduction.initialize_training_set(
                ntrain_SCM, enable_import=True, sampling=sampling)  # kwargs are not needed
            # If an exception is raised we will fall back to exact evaluation,
            # and thus we cannot use a distributed training set
            self.training_set.serialize_maximum_computations()
            self.SCM_reduction.training_set.serialize_maximum_computations()
            # Return
            return import_successful and import_successful_SCM

        # ERROR ANALYSIS: set the elements in the testing set.
        def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(
                self, ntest, enable_import, sampling, **kwargs)
            # Initialize testing set of SCM reduction
            assert "SCM" in kwargs
            ntest_SCM = kwargs["SCM"]
            import_successful_SCM = self.SCM_reduction.initialize_testing_set(ntest_SCM, enable_import, sampling)
            # kwargs are not needed
            # Return
            return import_successful and import_successful_SCM

        # Perform the offline phase of the reduced order model
        def offline(self):
            # Perform first the SCM offline phase, ...
            bak_first_mu = self.truth_problem.mu
            self.SCM_reduction.offline()
            # ..., and then call the parent method.
            self.truth_problem.set_mu(bak_first_mu)
            return DifferentialProblemReductionMethod_DerivedClass.offline(self)

        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        def error_analysis(self, N_generator=None, filename=None, **kwargs):
            # Perform first the SCM error analysis, ...
            if (
                "with_respect_to" not in kwargs
                # otherwise we assume the user was interested in computing the error w.r.t.
                # an exact stability factor, so he probably is not interested in the error analysis of SCM
                and "SCM" not in kwargs
                # otherwise we assume the user was interested in computing the error for a fixed number of SCM basis
                # functions, thus he has already carried out the error analysis of SCM
            ):
                SCM_N_generator = kwargs.pop("SCM_N_generator", None)
                self.SCM_reduction.error_analysis(SCM_N_generator, filename)
            # ..., and then call the parent method.
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N_generator, filename, **kwargs)

        def _init_error_analysis(self, **kwargs):
            # Replace stability factor computation, if needed
            self._replace_stability_factor_lower_bound_computation(**kwargs)
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_error_analysis(self, **kwargs)

        def _finalize_error_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_error_analysis(self, **kwargs)
            # Undo replacement of stability factor computation, if needed
            self._undo_replace_stability_factor_lower_bound_computation(**kwargs)

        # Compute the speedup of the reduced order approximation with respect to the full order one
        # over the testing set
        def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
            # Perform first the SCM speedup analysis, ...
            if (
                "with_respect_to" not in kwargs
                # otherwise we assume the user was interested in computing the speedup w.r.t.
                # an exact stability factor, so he probably is not interested in the speedup analysis of SCM
                and "SCM" not in kwargs
                # otherwise we assume the user was interested in computing the speedup for a fixed number of SCM basis
                # functions, thus he has already carried out the speedup analysis of SCM
            ):
                SCM_N_generator = kwargs.pop("SCM_N_generator", None)
                self.SCM_reduction.speedup_analysis(SCM_N_generator, filename)
            # ..., and then call the parent method.
            DifferentialProblemReductionMethod_DerivedClass.speedup_analysis(self, N_generator, filename, **kwargs)

        def _init_speedup_analysis(self, **kwargs):
            # Replace stability factor computation, if needed
            self._replace_stability_factor_lower_bound_computation(**kwargs)
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_speedup_analysis(self, **kwargs)

        def _finalize_speedup_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_speedup_analysis(self, **kwargs)
            # Undo replacement of stability factor computation, if needed
            self._undo_replace_stability_factor_lower_bound_computation(**kwargs)

        def _replace_stability_factor_lower_bound_computation(self, **kwargs):
            self._replace_stability_factor_lower_bound_computation__get_stability_factor_lower_bound__original = (
                self.reduced_problem.get_stability_factor_lower_bound)
            if "SCM" not in kwargs:
                if "with_respect_to" in kwargs:
                    assert inspect.isfunction(kwargs["with_respect_to"])
                    other_truth_problem = kwargs["with_respect_to"](self.truth_problem)
                    # Assume that the user wants to disable SCM and use the exact stability factor
                    self.replace_get_stability_factor_lower_bound = PatchInstanceMethod(
                        self.truth_problem,
                        "get_stability_factor_lower_bound",
                        lambda self_: other_truth_problem.get_stability_factor_lower_bound()
                    )
                    self.replace_get_stability_factor_lower_bound.patch()
                else:
                    self.replace_get_stability_factor_lower_bound = None
            else:
                assert isinstance(kwargs["SCM"], int)
                # Assume that the user wants to use SCM with a prescribed number of basis functions
                self.replace_get_stability_factor_lower_bound = PatchInstanceMethod(
                    self.truth_problem,
                    "get_stability_factor_lower_bound",
                    lambda self_: self_.SCM_approximation.get_stability_factor_lower_bound(kwargs["SCM"])
                )
                self.replace_get_stability_factor_lower_bound.patch()

        def _undo_replace_stability_factor_lower_bound_computation(self, **kwargs):
            if self.replace_get_stability_factor_lower_bound is not None:
                self.replace_get_stability_factor_lower_bound.unpatch()
            del self.replace_get_stability_factor_lower_bound

    # return value (a class) for the decorator
    return SCMDecoratedReductionMethod_Class
