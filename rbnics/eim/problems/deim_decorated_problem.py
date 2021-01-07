# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import inspect
from rbnics.backends import ParametrizedTensorFactory
from rbnics.eim.backends import OfflineOnlineBackend
from rbnics.eim.problems.eim_approximation import EIMApproximation as DEIMApproximation
from rbnics.eim.problems.time_dependent_eim_approximation import (
    TimeDependentEIMApproximation as TimeDependentDEIMApproximation)
from rbnics.eim.utils.decorators import DefineSymbolicParameters
from rbnics.utils.decorators import overload, PreserveClassName, ProblemDecoratorFor, tuple_of
from rbnics.utils.test import PatchInstanceMethod


def ExactDEIMAlgorithm(**kwargs):
    # Enable exact parametrized functions evaluation both offline and online
    from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
    kwargs["stages"] = ("offline", "online")
    return ExactParametrizedFunctions(**kwargs)


def DEIMDecoratedProblem(
    stages=("offline", "online"),
    basis_generation="POD",
    **decorator_kwargs
):
    from rbnics.eim.problems.deim import DEIM

    @ProblemDecoratorFor(DEIM, ExactAlgorithm=ExactDEIMAlgorithm, stages=stages, basis_generation=basis_generation)
    def DEIMDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):

        @DefineSymbolicParameters
        @PreserveClassName
        class DEIMDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for DEIM reduced problems
                self.DEIM_approximations = dict()  # from term to dict of DEIMApproximation
                self.non_DEIM_forms = dict()  # from term to dict of forms

                # Store value of N_DEIM passed to solve
                self._N_DEIM = None
                # Store values passed to decorator
                self._store_DEIM_stages(stages)
                # Avoid useless assignments
                self._update_N_DEIM__previous_kwargs = None

                # Generate offline online backend for current problem
                self.offline_online_backend = OfflineOnlineBackend(self.name())

            @overload(str)
            def _store_DEIM_stages(self, stage):
                assert stages != "offline", (
                    "This choice does not make any sense because it requires a DEIM offline stage"
                    + " which then is not used online")
                assert stages == "online"
                self._apply_DEIM_at_stages = (stages, )
                assert hasattr(self, "_apply_exact_evaluation_at_stages"), (
                    "Please apply @ExactParametrizedFunctions(\"offline\") below @DEIM(\"online\") decorator")
                assert self._apply_exact_evaluation_at_stages == ("offline", )

            @overload(tuple_of(str))
            def _store_DEIM_stages(self, stage):
                assert len(stages) in (1, 2)
                assert stages[0] in ("offline", "online")
                if len(stages) > 1:
                    assert stages[1] in ("offline", "online")
                    assert stages[0] != stages[1]
                self._apply_DEIM_at_stages = stages
                assert not hasattr(self, "_apply_exact_evaluation_at_stages"), (
                    "This choice does not make any sense because there is at least a stage for which"
                    + " both DEIM and ExactParametrizedFunctions are required")

            def _init_DEIM_approximations(self):
                # Preprocess each term in the affine expansions.
                # Note that this cannot be done in __init__, because operators may depend on self.mu,
                # which is not defined at __init__ time. Moreover, it cannot be done either by init,
                # because the init method is called by offline stage of the reduction method instance,
                # but we need DEIM approximations to be already set up at the time the reduction method
                # instance is built. Thus, we will call this method in the reduction method instance
                # constructor (having a safeguard in place to avoid repeated calls).
                assert (len(self.DEIM_approximations) == 0) == (len(self.non_DEIM_forms) == 0)
                if len(self.DEIM_approximations) == 0:  # initialize DEIM approximations only once
                    # Temporarily replace float parameters with symbols, so that we can detect if operators
                    # are parametrized
                    self.attach_symbolic_parameters()
                    # Loop over each term
                    for term in self.terms:
                        try:
                            forms = self.assemble_operator(term)
                        except ValueError:  # possibily raised e.g. because output computation is optional
                            pass
                        else:
                            self.DEIM_approximations[term] = dict()
                            self.non_DEIM_forms[term] = dict()
                            for (q, form_q) in enumerate(forms):
                                factory_form_q = ParametrizedTensorFactory(form_q)
                                if factory_form_q.is_parametrized():
                                    if factory_form_q.is_time_dependent():
                                        DEIMApproximationType = TimeDependentDEIMApproximation
                                    else:
                                        DEIMApproximationType = DEIMApproximation
                                    self.DEIM_approximations[term][q] = DEIMApproximationType(
                                        self, factory_form_q, os.path.join(self.name(), "deim", factory_form_q.name()),
                                        basis_generation)
                                else:
                                    self.non_DEIM_forms[term][q] = form_q
                    # Restore float parameters
                    self.detach_symbolic_parameters()

            def init(self):
                # Call parent's method (enforcing an empty parent call to _init_operators)
                self.disable_init_operators = PatchInstanceMethod(self, "_init_operators", lambda self_: None)
                # self.disable_init_operators may be shared between DEIM and exact evaluation
                self.disable_init_operators.patch()
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                self.disable_init_operators.unpatch()
                del self.disable_init_operators
                # Then, initialize DEIM operators
                self._init_operators_DEIM()

            def _init_operators_DEIM(self):
                # Initialize offline/online switch storage only once
                # (may be shared between DEIM and exact evaluation)
                OfflineOnlineClassMethod = self.offline_online_backend.OfflineOnlineClassMethod
                OfflineOnlineExpansionStorage = self.offline_online_backend.OfflineOnlineExpansionStorage
                OfflineOnlineExpansionStorageSize = self.offline_online_backend.OfflineOnlineExpansionStorageSize
                OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
                if not isinstance(self.Q, OfflineOnlineSwitch):
                    assert isinstance(self.Q, dict)
                    assert len(self.Q) == 0
                    self.Q = OfflineOnlineExpansionStorageSize()
                if not isinstance(self.operator, OfflineOnlineSwitch):
                    assert isinstance(self.operator, dict)
                    assert len(self.operator) == 0
                    self.operator = OfflineOnlineExpansionStorage(self, "OperatorExpansionStorage")
                if not isinstance(self.assemble_operator, OfflineOnlineSwitch):
                    assert inspect.ismethod(self.assemble_operator)
                    self.assemble_operator = OfflineOnlineClassMethod(self, "assemble_operator")
                if not isinstance(self.compute_theta, OfflineOnlineSwitch):
                    assert inspect.ismethod(self.compute_theta)
                    self.compute_theta = OfflineOnlineClassMethod(self, "compute_theta")
                # Setup offline/online switches
                former_stage = OfflineOnlineSwitch.get_current_stage()
                for stage_DEIM in self._apply_DEIM_at_stages:
                    OfflineOnlineSwitch.set_current_stage(stage_DEIM)
                    # Replace assemble_operator and compute_theta with DEIM computations
                    self.assemble_operator.attach(self._assemble_operator_DEIM,
                                                  lambda term: term in self.DEIM_approximations)
                    self.compute_theta.attach(self._compute_theta_DEIM,
                                              lambda term: term in self.DEIM_approximations)
                    # Setup offline/online operators storage with DEIM operators
                    self.operator.set_is_affine(True)
                    self._init_operators()
                    self.operator.unset_is_affine()
                # Restore former stage in offline/online switch storage
                OfflineOnlineSwitch.set_current_stage(former_stage)

            def _solve(self, **kwargs):
                self._update_N_DEIM(**kwargs)
                ParametrizedDifferentialProblem_DerivedClass._solve(self, **kwargs)

            def _update_N_DEIM(self, **kwargs):
                N_DEIM = kwargs.pop("DEIM", None)
                if N_DEIM != self._update_N_DEIM__previous_kwargs:
                    if N_DEIM is not None:
                        assert isinstance(N_DEIM, (dict, int))
                        if isinstance(N_DEIM, int):
                            N_DEIM_dict = dict()
                            for term in self.DEIM_approximations.keys():
                                N_DEIM_dict[term] = dict()
                                for q in self.DEIM_approximations[term]:
                                    N_DEIM_dict[term][q] = N_DEIM
                            self._N_DEIM = N_DEIM_dict
                        else:
                            self._N_DEIM = N_DEIM
                    else:
                        self._N_DEIM = None
                    self._update_N_DEIM__previous_kwargs = N_DEIM

            def _assemble_operator_DEIM(self, term):
                deim_forms = list()
                # Append forms computed with DEIM, if applicable
                for (_, deim_approximation) in self.DEIM_approximations[term].items():
                    deim_forms.extend(deim_approximation.basis_functions)
                # Append forms which did not require DEIM, if applicable
                for (_, non_deim_form) in self.non_DEIM_forms[term].items():
                    deim_forms.append(non_deim_form)
                return tuple(deim_forms)

            def _compute_theta_DEIM(self, term):
                original_thetas = ParametrizedDifferentialProblem_DerivedClass.compute_theta(self, term)
                deim_thetas = list()
                assert len(self.DEIM_approximations[term]) + len(self.non_DEIM_forms[term]) == len(original_thetas)
                if self._N_DEIM is not None:
                    assert term in self._N_DEIM
                    assert len(self.DEIM_approximations[term]) == len(self._N_DEIM[term])
                # Append forms computed with DEIM, if applicable
                for (q, deim_approximation) in self.DEIM_approximations[term].items():
                    N_DEIM = None
                    if self._N_DEIM is not None:
                        N_DEIM = self._N_DEIM[term][q]
                    deim_thetas_q = [v * original_thetas[q]
                                     for v in deim_approximation.compute_interpolated_theta(N_DEIM)]
                    deim_thetas.extend(deim_thetas_q)
                # Append forms which did not require DEIM, if applicable
                for q in self.non_DEIM_forms[term]:
                    deim_thetas.append(original_thetas[q])
                return tuple(deim_thetas)

            def _cache_key_from_kwargs(self, **kwargs):
                cache_key = ParametrizedDifferentialProblem_DerivedClass._cache_key_from_kwargs(self, **kwargs)
                # Change cache key depending on current stage
                OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
                if OfflineOnlineSwitch.get_current_stage() in self._apply_DEIM_at_stages:
                    # Append current stage to cache key
                    cache_key = cache_key + ("DEIM", )
                # Return
                return cache_key

        # return value (a class) for the decorator
        return DEIMDecoratedProblem_Class

    # return the decorator itself
    return DEIMDecoratedProblem_Decorator
