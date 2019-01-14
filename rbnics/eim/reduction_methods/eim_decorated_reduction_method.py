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

from numbers import Number
from rbnics.eim.problems import EIM
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.problems.time_dependent_eim_approximation import TimeDependentEIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.eim.reduction_methods.time_dependent_eim_approximation_reduction_method import TimeDependentEIMApproximationReductionMethod
from rbnics.utils.decorators import is_training_finished, PreserveClassName, ReductionMethodDecoratorFor, set_map_from_problem_to_training_status_off, set_map_from_problem_to_training_status_on

@ReductionMethodDecoratorFor(EIM)
def EIMDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    
    @PreserveClassName
    class EIMDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Initialize EIM approximations, if needed
            truth_problem._init_EIM_approximations()
            
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Storage for EIM reduction methods
            self.EIM_reductions = dict() # from coefficients to _EIMReductionMethod
            
            # Preprocess each term in the affine expansions
            for (coeff, EIM_approximation_coeff) in self.truth_problem.EIM_approximations.items():
                assert isinstance(EIM_approximation_coeff, (EIMApproximation, TimeDependentEIMApproximation))
                if isinstance(EIM_approximation_coeff, TimeDependentEIMApproximation):
                    EIMApproximationReductionMethodType = TimeDependentEIMApproximationReductionMethod
                else:
                    EIMApproximationReductionMethodType = EIMApproximationReductionMethod
                self.EIM_reductions[coeff] = EIMApproximationReductionMethodType(EIM_approximation_coeff)
            
        # OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_Nmax(self, Nmax, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            # Set Nmax of EIM reductions
            def setter(EIM_reduction, Nmax_EIM):
                EIM_reduction.set_Nmax(max(EIM_reduction.Nmax, Nmax_EIM)) # kwargs are not needed
            self._propagate_setter_from_kwargs_to_EIM_reductions(setter, int, **kwargs)
            
        # OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_tolerance(self, tol, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
            # Set tolerance of EIM reductions
            def setter(EIM_reduction, tol_EIM):
                EIM_reduction.set_tolerance(max(EIM_reduction.tol, tol_EIM)) # kwargs are not needed
            self._propagate_setter_from_kwargs_to_EIM_reductions(setter, Number, **kwargs)
            
        # OFFLINE: set the elements in the training set.
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
            # Since exact evaluation is required, we cannot use a distributed training set
            self.training_set.serialize_maximum_computations()
            # Initialize training set of EIM reductions
            def setter(EIM_reduction, ntrain_EIM):
                return EIM_reduction.initialize_training_set(ntrain_EIM, enable_import, sampling) # kwargs are not needed
            import_successful_EIM = self._propagate_setter_from_kwargs_to_EIM_reductions(setter, int, **kwargs)
            return import_successful and import_successful_EIM
            
        # ERROR ANALYSIS: set the elements in the testing set.
        def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
            # Initialize testing set of EIM reductions
            def setter(EIM_reduction, ntest_EIM):
                return EIM_reduction.initialize_testing_set(ntest_EIM, enable_import, sampling) # kwargs are not needed
            import_successful_EIM = self._propagate_setter_from_kwargs_to_EIM_reductions(setter, int, **kwargs)
            return import_successful and import_successful_EIM
            
        def _propagate_setter_from_kwargs_to_EIM_reductions(self, setter, Type, **kwargs):
            assert "EIM" in kwargs
            kwarg_EIM = kwargs["EIM"]
            return_value = True # will be either a bool or None
            if isinstance(kwarg_EIM, dict):
                for term in self.truth_problem.separated_forms:
                    if sum([len(form.coefficients) for form in self.truth_problem.separated_forms[term]]) > 0:
                        assert term in kwarg_EIM, "Please provide a value for term " + str(term)
                        assert isinstance(kwarg_EIM[term], (int, tuple))
                        if isinstance(kwarg_EIM[term], int):
                            kwarg_EIM[term] = [kwarg_EIM[term]]*len(self.truth_problem.separated_forms[term])
                        else:
                            assert len(self.truth_problem.separated_forms[term]) == len(kwarg_EIM[term])
                        for (form, kwarg_EIM_form) in zip(self.truth_problem.separated_forms[term], kwarg_EIM[term]):
                            for addend in form.coefficients:
                                for factor in addend:
                                    assert factor in self.EIM_reductions
                                    assert isinstance(kwarg_EIM_form, Type)
                                    current_return_value = setter(self.EIM_reductions[factor], kwarg_EIM_form)
                                    return_value = current_return_value and return_value
            else:
                assert isinstance(kwarg_EIM, Type)
                for (coeff, EIM_reduction_coeff) in self.EIM_reductions.items():
                    current_return_value = setter(EIM_reduction_coeff, kwarg_EIM)
                    return_value = current_return_value and return_value
            return return_value # an "and" with a None results in None, so this method returns only if necessary
            
        # Perform the offline phase of the reduced order model
        def offline(self):
            if "offline" not in self.truth_problem._apply_EIM_at_stages:
                assert hasattr(self.truth_problem, "_apply_exact_evaluation_at_stages"), "Please use @ExactParametrizedFunctions(\"offline\")"
                assert "offline" in self.truth_problem._apply_exact_evaluation_at_stages, "Please use @ExactParametrizedFunctions(\"offline\")"
            lifting_mu = self.truth_problem.mu
            for (coeff, EIM_reduction_coeff) in self.EIM_reductions.items():
                EIM_reduction_coeff.offline()
            self.truth_problem.set_mu(lifting_mu)
            return DifferentialProblemReductionMethod_DerivedClass.offline(self)
            
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        def error_analysis(self, N_generator=None, filename=None, **kwargs):
            # Perform first the EIM error analysis, ...
            if (
                "with_respect_to" not in kwargs # otherwise we assume the user was interested in computing the error w.r.t.
                                                # an exact parametrized functions,
                                                # so he probably is not interested in the error analysis of EIM
                    and
                (
                    "EIM" not in kwargs         # otherwise we assume the user was interested in computing the error for a fixed number of EIM basis
                                                # functions, thus he has already carried out the error analysis of EIM
                        or
                    ("EIM" in kwargs and kwargs["EIM"] is not None) # shorthand to disable EIM error analysis
                )
            ):
                EIM_N_generator = kwargs.pop("EIM_N_generator", None)
                assert is_training_finished(self.truth_problem)
                set_map_from_problem_to_training_status_off(self.truth_problem)
                for (coeff, EIM_reduction_coeff) in self.EIM_reductions.items():
                    EIM_reduction_coeff.error_analysis(EIM_N_generator, filename)
                set_map_from_problem_to_training_status_on(self.truth_problem)
            # ..., and then call the parent method.
            if "EIM" in kwargs and kwargs["EIM"] is None:
                del kwargs["EIM"]
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N_generator, filename, **kwargs)
            
        # Compute the speedup of the reduced order approximation with respect to the full order one
        # over the testing set
        def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
            # Perform first the EIM speedup analysis, ...
            if (
                "with_respect_to" not in kwargs # otherwise we assume the user was interested in computing the speedup w.r.t.
                                                # an exact parametrized functions,
                                                # so he probably is not interested in the speedup analysis of EIM
                    and
                (
                    "EIM" not in kwargs         # otherwise we assume the user was interested in computing the speedup for a fixed number of EIM basis
                                                # functions, thus he has already carried out the speedup analysis of EIM
                        or
                    ("EIM" in kwargs and kwargs["EIM"] is not None) # shorthand to disable EIM error analysis
                )
            ):
                EIM_N_generator = kwargs.pop("EIM_N_generator", None)
                assert is_training_finished(self.truth_problem)
                set_map_from_problem_to_training_status_off(self.truth_problem)
                for (coeff, EIM_reduction_coeff) in self.EIM_reductions.items():
                    EIM_reduction_coeff.speedup_analysis(EIM_N_generator, filename)
                set_map_from_problem_to_training_status_on(self.truth_problem)
            # ..., and then call the parent method.
            if "EIM" in kwargs and kwargs["EIM"] is None:
                del kwargs["EIM"]
            DifferentialProblemReductionMethod_DerivedClass.speedup_analysis(self, N_generator, filename, **kwargs)
        
    # return value (a class) for the decorator
    return EIMDecoratedReductionMethod_Class
