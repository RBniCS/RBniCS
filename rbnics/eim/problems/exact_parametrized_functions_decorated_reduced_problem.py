# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import inspect
from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
from rbnics.eim.utils.decorators import (StoreMapFromBasisFunctionsToReducedProblem,
                                         StoreMapFromEachBasisFunctionToComponentAndIndex,
                                         StoreMapFromRieszStorageToReducedProblem)
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from rbnics.utils.test import PatchInstanceMethod


@ReducedProblemDecoratorFor(ExactParametrizedFunctions)
def ExactParametrizedFunctionsDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    def _AlsoDecorateErrorEstimationOperators(ParametrizedReducedDifferentialProblem_DecoratedClass):
        if hasattr(ParametrizedReducedDifferentialProblem_DecoratedClass, "assemble_error_estimation_operators"):

            @StoreMapFromEachBasisFunctionToComponentAndIndex
            @StoreMapFromRieszStorageToReducedProblem
            @PreserveClassName
            class _AlsoDecorateErrorEstimationOperators_Class(ParametrizedReducedDifferentialProblem_DecoratedClass):

                def init(self, current_stage="online"):
                    # self.disable_init_error_estimation_operators may be shared between EIM/DEIM and exact evaluation
                    has_disable_init_error_estimation_operators = hasattr(
                        self, "disable_init_error_estimation_operators")
                    # Call parent's method (enforcing an empty parent call to _init_error_estimation_operators)
                    if not has_disable_init_error_estimation_operators:
                        self.disable_init_error_estimation_operators = PatchInstanceMethod(
                            self, "_init_error_estimation_operators", lambda self_, current_stage="online": None)
                    self.disable_init_error_estimation_operators.patch()
                    ParametrizedReducedDifferentialProblem_DecoratedClass.init(self, current_stage)
                    self.disable_init_error_estimation_operators.unpatch()
                    if not has_disable_init_error_estimation_operators:
                        del self.disable_init_error_estimation_operators
                    # Then, initialize error estimation operators associated to exact operators
                    self._init_error_estimation_operators_exact(current_stage)

                def _init_error_estimation_operators_exact(self, current_stage="online"):
                    # Initialize offline/online switch storage only once (may be shared between EIM/DEIM
                    # and exact evaluation)
                    OfflineOnlineExpansionStorage = self.offline_online_backend.OfflineOnlineExpansionStorage
                    OfflineOnlineRieszSolver = self.offline_online_backend.OfflineOnlineRieszSolver
                    OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
                    if not isinstance(self.riesz, OfflineOnlineSwitch):
                        assert isinstance(self.riesz, dict)
                        assert len(self.riesz) == 0
                        self.riesz = OfflineOnlineExpansionStorage(self, "RieszExpansionStorage")
                    if not isinstance(self.error_estimation_operator, OfflineOnlineSwitch):
                        assert isinstance(self.error_estimation_operator, dict)
                        assert len(self.error_estimation_operator) == 0
                        self.error_estimation_operator = OfflineOnlineExpansionStorage(
                            self, "ErrorEstimationOperatorExpansionStorage")
                    if not isinstance(self.RieszSolver, OfflineOnlineSwitch):
                        assert inspect.isclass(self.RieszSolver)
                        self.RieszSolver = OfflineOnlineRieszSolver()
                    # Setup offline/online operators storage with exact operators
                    assert current_stage in ("online", "offline")
                    apply_exact_evaluation_at_stages = self.truth_problem._apply_exact_evaluation_at_stages
                    if current_stage == "online":
                        apply_exact_evaluation_at_stages = (
                            "online", ) if "online" in apply_exact_evaluation_at_stages else ()
                    for stage_exact in apply_exact_evaluation_at_stages:
                        OfflineOnlineSwitch.set_current_stage(stage_exact)
                        self.riesz.set_is_affine(False)
                        self.error_estimation_operator.set_is_affine(False)
                        self.RieszSolver.set_is_affine(False)
                        self._init_error_estimation_operators(current_stage)
                        self.riesz.unset_is_affine()
                        self.error_estimation_operator.unset_is_affine()
                        self.RieszSolver.unset_is_affine()
                    # Update current stage in offline/online switch
                    OfflineOnlineSwitch.set_current_stage(current_stage)

                def build_error_estimation_operators(self, current_stage="offline"):
                    # self.disable_build_error_estimation_operators may be shared between EIM/DEIM and exact evaluation
                    has_disable_build_error_estimation_operators = hasattr(
                        self, "disable_build_error_estimation_operators")
                    # Call parent's method (enforcing an empty parent call to _build_error_estimation_operators)
                    if not has_disable_build_error_estimation_operators:
                        self.disable_build_error_estimation_operators = PatchInstanceMethod(
                            self, "_build_error_estimation_operators", lambda self_, current_stage="offline": None)
                    self.disable_build_error_estimation_operators.patch()
                    ParametrizedReducedDifferentialProblem_DecoratedClass.build_error_estimation_operators(
                        self, current_stage)
                    self.disable_build_error_estimation_operators.unpatch()
                    if not has_disable_build_error_estimation_operators:
                        del self.disable_build_error_estimation_operators
                    # Then, build exact operators
                    self._build_error_estimation_operators_exact(current_stage)

                def _build_error_estimation_operators_exact(self, current_stage="offline"):
                    # Build offline/online operators storage from exact operators
                    OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
                    assert current_stage == "offline"
                    for stage_exact in self.truth_problem._apply_exact_evaluation_at_stages:
                        OfflineOnlineSwitch.set_current_stage(stage_exact)
                        self._build_error_estimation_operators(current_stage)
                        OfflineOnlineSwitch.set_current_stage(current_stage)

            return _AlsoDecorateErrorEstimationOperators_Class
        else:
            return ParametrizedReducedDifferentialProblem_DecoratedClass

    @_AlsoDecorateErrorEstimationOperators
    @StoreMapFromBasisFunctionsToReducedProblem
    @PreserveClassName
    class ExactParametrizedFunctionsDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):

        def __init__(self, truth_problem, **kwargs):
            # Call parent's method
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Copy offline online backend for current problem
            self.offline_online_backend = truth_problem.offline_online_backend

        def init(self, current_stage="online"):
            # self.disable_init_operators may be shared between EIM/DEIM and exact evaluation
            has_disable_init_operators = hasattr(self, "disable_init_operators")
            # Call parent's method (enforcing an empty parent call to _init_operators)
            if not has_disable_init_operators:
                self.disable_init_operators = PatchInstanceMethod(
                    self, "_init_operators", lambda self_, current_stage="online": None)
            self.disable_init_operators.patch()
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            self.disable_init_operators.unpatch()
            if not has_disable_init_operators:
                del self.disable_init_operators
            # Then, initialize exact operators
            self._init_operators_exact(current_stage)

        def _init_operators_exact(self, current_stage="online"):
            # Initialize offline/online switch storage only once (may be shared between EIM/DEIM and exact evaluation)
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
            # Setup offline/online operators storage with exact operators
            assert current_stage in ("online", "offline")
            apply_exact_evaluation_at_stages = self.truth_problem._apply_exact_evaluation_at_stages
            if current_stage == "online":
                apply_exact_evaluation_at_stages = ("online", ) if "online" in apply_exact_evaluation_at_stages else ()
            for stage_exact in apply_exact_evaluation_at_stages:
                OfflineOnlineSwitch.set_current_stage(stage_exact)
                self.operator.set_is_affine(False)
                self._init_operators(current_stage)
                self.operator.unset_is_affine()
            # Update current stage in offline/online switch
            OfflineOnlineSwitch.set_current_stage(current_stage)

        def build_reduced_operators(self, current_stage="offline"):
            # self.disable_build_reduced_operators may be shared between EIM/DEIM and exact evaluation
            has_disable_build_reduced_operators = hasattr(self, "disable_build_reduced_operators")
            # Call parent's method (enforcing an empty parent call to _build_reduced_operators)
            if not has_disable_build_reduced_operators:
                self.disable_build_reduced_operators = PatchInstanceMethod(
                    self, "_build_reduced_operators", lambda self_, current_stage="offline": None)
            self.disable_build_reduced_operators.patch()
            ParametrizedReducedDifferentialProblem_DerivedClass.build_reduced_operators(self, current_stage)
            self.disable_build_reduced_operators.unpatch()
            if not has_disable_build_reduced_operators:
                del self.disable_build_reduced_operators
            # Then, build exact operators
            self._build_reduced_operators_exact(current_stage)

        def _build_reduced_operators_exact(self, current_stage="offline"):
            # Build offline/online operators storage from exact operators
            OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
            assert current_stage == "offline"
            for stage_exact in self.truth_problem._apply_exact_evaluation_at_stages:
                OfflineOnlineSwitch.set_current_stage(stage_exact)
                self._build_reduced_operators(current_stage)
            # Update current stage in offline/online switch
            OfflineOnlineSwitch.set_current_stage(current_stage)

        def _cache_key_from_N_and_kwargs(self, N, **kwargs):
            if len(self.truth_problem._apply_exact_evaluation_at_stages) == 1:
                # uses EIM/DEIM online and exact evaluation offline
                cache_key = ParametrizedReducedDifferentialProblem_DerivedClass._cache_key_from_N_and_kwargs(
                    self, N, **kwargs)
                # Append current stage to cache key
                OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
                cache_key = cache_key + (OfflineOnlineSwitch.get_current_stage(), )
                # Return
                return cache_key
            else:
                return ParametrizedReducedDifferentialProblem_DerivedClass._cache_key_from_N_and_kwargs(
                    self, N, **kwargs)

    # return value (a class) for the decorator
    return ExactParametrizedFunctionsDecoratedReducedProblem_Class
