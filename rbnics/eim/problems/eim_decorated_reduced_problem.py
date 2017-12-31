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

from rbnics.backends import AffineExpansionStorage
from rbnics.eim.problems.eim import EIM
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor

@ReducedProblemDecoratorFor(EIM)
def EIMDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    def _AlsoDecorateErrorEstimationOperators(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass, "assemble_error_estimation_operators"):
        
            @PreserveClassName
            class _AlsoDecorateErrorEstimationOperators_Class(ReducedParametrizedProblem_DecoratedClass):
                
                def compute_riesz(self, term):
                    # Temporarily swap truth problem exact operator with EIM operators
                    if "offline" not in self.truth_problem_for_EIM._apply_EIM_at_stages:
                        self._set_truth_problem_operator_with_EIM(term)
                    # Call Parent
                    ParametrizedReducedDifferentialProblem_DerivedClass.compute_riesz(self, term)
                    # Restore truth problem exact operator
                    if "offline" not in self.truth_problem_for_EIM._apply_EIM_at_stages:
                        self._set_truth_problem_operator_without_EIM(term)
                    
                def assemble_error_estimation_operators(self, term, current_stage="online"):
                    if current_stage == "offline":
                        if term in self.truth_problem_for_EIM.separated_forms.keys():
                            if "offline" not in self.truth_problem_for_EIM._apply_EIM_at_stages:
                                # Temporarily swap truth problem exact operator with EIM operators
                                self._set_truth_problem_operator_with_EIM(term)
                                # Call Parent
                                operator = ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, current_stage)
                                # Restore truth problem exact operator
                                self._set_truth_problem_operator_without_EIM(term)
                                # Return
                                return operator
                            else:
                                return ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, current_stage)
                        else:
                            return ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, current_stage)
                    else:
                        return ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, current_stage)
                                
            return _AlsoDecorateErrorEstimationOperators_Class
        else:
            return ReducedParametrizedProblem_DecoratedClass
    
    @_AlsoDecorateErrorEstimationOperators
    @PreserveClassName
    class EIMDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Store the truth problem which should be updated for EIM
            self.truth_problem_for_EIM = self.truth_problem
            # ... this makes sure that, in case self.truth_problem is replaced (because of multilevel reduction)
            #     the correct problem is called for what concerns EIM computations
            # Storage for truth problem operators after EIM if they are not used during the offline solve because
            # EIM is requested only online
            self.truth_problem_operator_with_EIM = dict()
            self.truth_problem_operator_without_EIM = dict()
            self.truth_problem_Q_with_EIM = dict()
            self.truth_problem_Q_without_EIM = dict()
            
        def _init_operators(self, current_stage="online"):
            # Fill in truth operators with and without EIM, and temporarily swap truth problem exact operator
            # with EIM operators
            if (
                "offline" not in self.truth_problem_for_EIM._apply_EIM_at_stages
                    and
                current_stage == "offline"
            ):
                for term in self.truth_problem_for_EIM.separated_forms.keys():
                    self.truth_problem_operator_with_EIM[term] = AffineExpansionStorage(self.truth_problem_for_EIM._assemble_operator_EIM(term))
                    self.truth_problem_operator_without_EIM[term] = self.truth_problem_for_EIM.operator[term]
                    self.truth_problem_Q_with_EIM[term] = len(self.truth_problem_operator_with_EIM[term])
                    self.truth_problem_Q_without_EIM[term] = self.truth_problem_for_EIM.Q[term]
                    self.Q[term] = self.truth_problem_Q_with_EIM[term]
                    self._set_truth_problem_operator_with_EIM(term)
            # Call Parent
            ParametrizedReducedDifferentialProblem_DerivedClass._init_operators(self, current_stage)
            # Restore truth problem exact operator
            if (
                "offline" not in self.truth_problem_for_EIM._apply_EIM_at_stages
                    and
                current_stage == "offline"
            ):
                for term in self.truth_problem_for_EIM.separated_forms.keys():
                    self._set_truth_problem_operator_without_EIM(term)
                    
        def _set_truth_problem_operator_with_EIM(self, term):
            self.truth_problem_for_EIM.operator[term] = self.truth_problem_operator_with_EIM[term]
            self.truth_problem_for_EIM.Q[term] = self.truth_problem_Q_with_EIM[term]
            
        def _set_truth_problem_operator_without_EIM(self, term):
            self.truth_problem_for_EIM.operator[term] = self.truth_problem_operator_without_EIM[term]
            self.truth_problem_for_EIM.Q[term] = self.truth_problem_Q_without_EIM[term]
            
        def _solve(self, N, **kwargs):
            self._update_N_EIM(**kwargs)
            ParametrizedReducedDifferentialProblem_DerivedClass._solve(self, N, **kwargs)
            
        def _update_N_EIM(self, **kwargs):
            self.truth_problem_for_EIM._update_N_EIM(**kwargs)
            
        def assemble_operator(self, term, current_stage="online"):
            if current_stage == "offline":
                if term in self.truth_problem_for_EIM.separated_forms.keys():
                    if "offline" not in self.truth_problem_for_EIM._apply_EIM_at_stages:
                        # Temporarily swap truth problem exact operator with EIM operators
                        self._set_truth_problem_operator_with_EIM(term)
                        # Call Parent
                        operator = ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
                        # Restore truth problem exact operator
                        self._set_truth_problem_operator_without_EIM(term)
                        # Return
                        return operator
                    else:
                        return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
                else:
                    return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
            else:
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
        
        def compute_theta(self, term):
            if term in self.truth_problem_for_EIM.separated_forms.keys():
                return self.truth_problem_for_EIM._compute_theta_EIM(term)
            else:
                return self.truth_problem_for_EIM.compute_theta(term)
        
    # return value (a class) for the decorator
    return EIMDecoratedReducedProblem_Class
