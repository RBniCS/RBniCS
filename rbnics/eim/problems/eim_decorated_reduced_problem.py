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

import inspect
from rbnics.eim.problems.eim import EIM
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from rbnics.utils.test import PatchInstanceMethod

@ReducedProblemDecoratorFor(EIM)
def EIMDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    def _AlsoDecorateErrorEstimationOperators(ParametrizedReducedDifferentialProblem_DecoratedClass):
        if hasattr(ParametrizedReducedDifferentialProblem_DecoratedClass, "assemble_error_estimation_operators"):
        
            @PreserveClassName
            class _AlsoDecorateErrorEstimationOperators_Class(ParametrizedReducedDifferentialProblem_DecoratedClass):
                
                def init(self, current_stage="online"):
                    # Call parent's method (enforcing an empty parent call to _init_error_estimation_operators)
                    self.disable_init_error_estimation_operators = PatchInstanceMethod(self, "_init_error_estimation_operators", lambda self_, current_stage="online": None) # may be shared between EIM and exact evaluation
                    self.disable_init_error_estimation_operators.patch()
                    ParametrizedReducedDifferentialProblem_DecoratedClass.init(self, current_stage)
                    self.disable_init_error_estimation_operators.unpatch()
                    del self.disable_init_error_estimation_operators
                    # Then, initialize error estimation operators associated to EIM operators
                    self._init_error_estimation_operators_EIM(current_stage)
                            
                def _init_error_estimation_operators_EIM(self, current_stage="online"):
                    # Initialize offline/online switch storage only once (may be shared between EIM and exact evaluation)
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
                        self.error_estimation_operator = OfflineOnlineExpansionStorage(self, "ErrorEstimationOperatorExpansionStorage")
                    if not isinstance(self.RieszSolver, OfflineOnlineSwitch):
                        assert inspect.isclass(self.RieszSolver)
                        self.RieszSolver = OfflineOnlineRieszSolver()
                    # Setup offline/online operators storage with EIM operators
                    assert current_stage in ("online", "offline")
                    apply_EIM_at_stages = self.truth_problem._apply_EIM_at_stages
                    if current_stage == "online":
                        apply_EIM_at_stages = ("online", ) if "online" in apply_EIM_at_stages else ()
                    for stage_EIM in apply_EIM_at_stages:
                        OfflineOnlineSwitch.set_current_stage(stage_EIM)
                        self.riesz.set_is_affine(True)
                        self.error_estimation_operator.set_is_affine(True)
                        self.RieszSolver.set_is_affine(True)
                        self._init_error_estimation_operators(current_stage)
                        self.riesz.unset_is_affine()
                        self.error_estimation_operator.unset_is_affine()
                        self.RieszSolver.unset_is_affine()
                    # Update current stage in offline/online switch
                    OfflineOnlineSwitch.set_current_stage(current_stage)
                    
                def build_error_estimation_operators(self, current_stage="offline"):
                    # Call parent's method (enforcing an empty parent call to _build_error_estimation_operators)
                    self.disable_build_error_estimation_operators = PatchInstanceMethod(self, "_build_error_estimation_operators", lambda self_, current_stage="offline": None) # may be shared between EIM and exact evaluation
                    self.disable_build_error_estimation_operators.patch()
                    ParametrizedReducedDifferentialProblem_DecoratedClass.build_error_estimation_operators(self, current_stage)
                    self.disable_build_error_estimation_operators.unpatch()
                    del self.disable_build_error_estimation_operators
                    # Then, build error estimators associated to EIM operators
                    self._build_error_estimation_operators_EIM(current_stage)
                    
                def _build_error_estimation_operators_EIM(self, current_stage="offline"):
                    # Build offline/online error estimators storage from EIM operators
                    OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
                    assert current_stage == "offline"
                    for stage_EIM in self.truth_problem._apply_EIM_at_stages:
                        OfflineOnlineSwitch.set_current_stage(stage_EIM)
                        self._build_error_estimation_operators(current_stage)
                        OfflineOnlineSwitch.set_current_stage(current_stage)
                    
            return _AlsoDecorateErrorEstimationOperators_Class
        else:
            return ParametrizedReducedDifferentialProblem_DecoratedClass
    
    @_AlsoDecorateErrorEstimationOperators
    @PreserveClassName
    class EIMDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        def __init__(self, truth_problem, **kwargs):
            # Call parent's method
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Copy offline online backend for current problem
            self.offline_online_backend = truth_problem.offline_online_backend
            
        def init(self, current_stage="online"):
            # Call parent's method (enforcing an empty parent call to _init_operators)
            self.disable_init_operators = PatchInstanceMethod(self, "_init_operators", lambda self_, current_stage="online": None) # may be shared between EIM and exact evaluation
            self.disable_init_operators.patch()
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            self.disable_init_operators.unpatch()
            del self.disable_init_operators
            # Then, initialize EIM operators
            self._init_operators_EIM(current_stage)
            
        def _init_operators_EIM(self, current_stage="online"):
            # Initialize offline/online switch storage only once (may be shared between EIM and exact evaluation)
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
            # Setup offline/online operators storage with EIM operators
            assert current_stage in ("online", "offline")
            apply_EIM_at_stages = self.truth_problem._apply_EIM_at_stages
            if current_stage == "online":
                apply_EIM_at_stages = ("online", ) if "online" in apply_EIM_at_stages else ()
            for stage_EIM in apply_EIM_at_stages:
                OfflineOnlineSwitch.set_current_stage(stage_EIM)
                self.operator.set_is_affine(True)
                self._init_operators(current_stage)
                self.operator.unset_is_affine()
            # Update current stage in offline/online switch
            OfflineOnlineSwitch.set_current_stage(current_stage)
                
        def _solve(self, N, **kwargs):
            self._update_N_EIM(**kwargs)
            ParametrizedReducedDifferentialProblem_DerivedClass._solve(self, N, **kwargs)
            
        def _update_N_EIM(self, **kwargs):
            self.truth_problem._update_N_EIM(**kwargs)
            
        def build_reduced_operators(self, current_stage="offline"):
            # Call parent's method (enforcing an empty parent call to _build_reduced_operators)
            self.disable_build_reduced_operators = PatchInstanceMethod(self, "_build_reduced_operators", lambda self_, current_stage="offline": None) # may be shared between EIM and exact evaluation
            self.disable_build_reduced_operators.patch()
            ParametrizedReducedDifferentialProblem_DerivedClass.build_reduced_operators(self, current_stage)
            self.disable_build_reduced_operators.unpatch()
            del self.disable_build_reduced_operators
            # Then, build EIM operators
            self._build_reduced_operators_EIM(current_stage)
            
        def _build_reduced_operators_EIM(self, current_stage="offline"):
            # Build offline/online operators storage from EIM operators
            OfflineOnlineSwitch = self.offline_online_backend.OfflineOnlineSwitch
            assert current_stage == "offline"
            for stage_EIM in self.truth_problem._apply_EIM_at_stages:
                OfflineOnlineSwitch.set_current_stage(stage_EIM)
                self._build_reduced_operators(current_stage)
            # Update current stage in offline/online switch
            OfflineOnlineSwitch.set_current_stage(current_stage)
                
    # return value (a class) for the decorator
    return EIMDecoratedReducedProblem_Class
