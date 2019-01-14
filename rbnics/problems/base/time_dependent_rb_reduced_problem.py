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

from rbnics.backends import product, sum, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from rbnics.problems.base.rb_reduced_problem import RBReducedProblem
from rbnics.problems.base.time_dependent_reduced_problem import TimeDependentReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators

@RequiredBaseDecorators(RBReducedProblem, TimeDependentReducedProblem)
def TimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class TimeDependentRBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
    
        # Default initialization of members.
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Storage related to error estimation for initial condition
            self.initial_condition_product = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components)
        
        def _init_error_estimation_operators(self, current_stage="online"):
            ParametrizedReducedDifferentialProblem_DerivedClass._init_error_estimation_operators(self, current_stage)
            # Also initialize data structures related to initial condition error estimation
            if len(self.components) > 1:
                initial_condition_product = dict()
                for component in self.components:
                    if self.initial_condition[component] and not self.initial_condition_is_homogeneous[component]:
                        initial_condition_product[component, component] = OnlineAffineExpansionStorage(self.Q_ic[component], self.Q_ic[component])
                        assert current_stage in ("online", "offline")
                        if current_stage == "online":
                            self.assemble_error_estimation_operators(("initial_condition_" + component, "initial_condition_" + component), "online")
                        elif current_stage == "offline":
                            pass # Nothing else to be done
                        else:
                            raise ValueError("Invalid stage in _init_error_estimation_operators().")
                if len(initial_condition_product) > 0:
                    self.initial_condition_product = initial_condition_product
            else:
                if self.initial_condition and not self.initial_condition_is_homogeneous:
                    self.initial_condition_product = OnlineAffineExpansionStorage(self.Q_ic, self.Q_ic)
                    assert current_stage in ("online", "offline")
                    if current_stage == "online":
                        self.assemble_error_estimation_operators(("initial_condition", "initial_condition"), "online")
                    elif current_stage == "offline":
                        pass # Nothing else to be done
                    else:
                        raise ValueError("Invalid stage in _init_error_estimation_operators().")
                    
        # Build operators for error estimation
        def build_error_estimation_operators(self, current_stage="offline"):
            ParametrizedReducedDifferentialProblem_DerivedClass.build_error_estimation_operators(self, current_stage)
            # Initial condition
            self._build_reduced_initial_condition_error_estimation(current_stage)
            
        def _build_reduced_initial_condition_error_estimation(self, current_stage="offline"):
            # Assemble initial condition product error estimation operator
            if len(self.components) > 1:
                for component in self.components:
                    if self.initial_condition[component] and not self.initial_condition_is_homogeneous[component]:
                        self.assemble_error_estimation_operators(("initial_condition_" + component, "initial_condition_" + component), current_stage)
            else:
                if self.initial_condition and not self.initial_condition_is_homogeneous:
                    self.assemble_error_estimation_operators(("initial_condition", "initial_condition"), current_stage)
        
        # Assemble operators for error estimation
        def assemble_error_estimation_operators(self, term, current_stage="online"):
            if term[0].startswith("initial_condition") and term[1].startswith("initial_condition"):
                component0 = term[0].replace("initial_condition", "").replace("_", "")
                component1 = term[1].replace("initial_condition", "").replace("_", "")
                assert current_stage in ("online", "offline")
                if current_stage == "online": # load from file
                    assert (component0 != "") == (component1 != "")
                    if component0 != "":
                        assert component0 in self.components
                        assert component1 in self.components
                        self.initial_condition_product[component0, component1].load(self.folder["error_estimation"], "initial_condition_product_" + component0 + "_" + component1)
                        return self.initial_condition_product[component0, component1]
                    else:
                        assert len(self.components) == 1
                        self.initial_condition_product.load(self.folder["error_estimation"], "initial_condition_product")
                        return self.initial_condition_product
                elif current_stage == "offline":
                    inner_product = self.truth_problem._combined_projection_inner_product
                    assert (component0 != "") == (component1 != "")
                    if component0 != "":
                        for q0 in range(self.Q_ic[component0]):
                            for q1 in range(self.Q_ic[component1]):
                                self.initial_condition_product[component0, component1][q0, q1] = transpose(self.truth_problem.initial_condition[component0][q0])*inner_product*self.truth_problem.initial_condition[component1][q1]
                        self.initial_condition_product[component0, component1].save(self.folder["error_estimation"], "initial_condition_product_" + component0 + "_" + component1)
                        return self.initial_condition_product[component0, component1]
                    else:
                        assert len(self.components) == 1
                        for q0 in range(self.Q_ic):
                            for q1 in range(self.Q_ic):
                                self.initial_condition_product[q0, q1] = transpose(self.truth_problem.initial_condition[q0])*inner_product*self.truth_problem.initial_condition[q1]
                        self.initial_condition_product.save(self.folder["error_estimation"], "initial_condition_product")
                        return self.initial_condition_product
                else:
                    raise ValueError("Invalid stage in assemble_error_estimation_operators().")
            else:
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_error_estimation_operators(self, term, current_stage)
                
        # Return the error bound for the initial condition
        def get_initial_error_estimate_squared(self):
            self._solution = self._solution_over_time[0]
            N = self._solution.N
            
            at_least_one_non_homogeneous_initial_condition = False
            
            addend_0 = 0.
            addend_1_right = OnlineFunction(N)
            
            if len(self.components) > 1:
                for component in self.components:
                    if self.initial_condition[component] and not self.initial_condition_is_homogeneous[component]:
                        at_least_one_non_homogeneous_initial_condition = True
                        theta_ic_component = self.compute_theta("initial_condition_" + component)
                        addend_0 += sum(product(theta_ic_component, self.initial_condition_product[component, component], theta_ic_component))
                        addend_1_right += sum(product(theta_ic_component, self.initial_condition[:N]))
            else:
                if self.initial_condition and not self.initial_condition_is_homogeneous:
                    at_least_one_non_homogeneous_initial_condition = True
                    theta_ic = self.compute_theta("initial_condition")
                    addend_0 += sum(product(theta_ic, self.initial_condition_product, theta_ic))
                    addend_1_right += sum(product(theta_ic, self.initial_condition[:N]))
            
            if at_least_one_non_homogeneous_initial_condition:
                inner_product_N = self._combined_projection_inner_product[:N, :N]
                addend_1_left = self._solution
                addend_2 = transpose(self._solution)*inner_product_N*self._solution
                return addend_0 - 2.0*(transpose(addend_1_left)*addend_1_right) + addend_2
            else:
                return 0.
                
    # return value (a class) for the decorator
    return TimeDependentRBReducedProblem_Class
