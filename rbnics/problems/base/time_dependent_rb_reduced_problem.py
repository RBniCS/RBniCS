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

from rbnics.backends import transpose
from rbnics.backends.online import OnlineAffineExpansionStorage
from rbnics.problems.base.rb_reduced_problem import RBReducedProblem
from rbnics.problems.base.time_dependent_reduced_problem import TimeDependentReducedProblem
from rbnics.utils.decorators import apply_decorator_only_once, Extends, override

@apply_decorator_only_once
def TimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    TimeDependentRBReducedProblem_Base = TimeDependentReducedProblem(RBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass))
    
    @Extends(TimeDependentRBReducedProblem_Base, preserve_class_name=True)
    class TimeDependentRBReducedProblem_Class(TimeDependentRBReducedProblem_Base):
    
        ## Default initialization of members.
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            TimeDependentRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)
            
            # Storage related to error estimation for initial condition
            self.initial_condition_product = None # will be of class OnlineAffineExpansionStorage
        
        def _init_error_estimation_operators(self, current_stage="online"):
            TimeDependentRBReducedProblem_Base._init_error_estimation_operators(self, current_stage)
            # Also initialize data structures related to initial condition error estimation
            if not self.initial_condition_is_homogeneous:
                assert current_stage in ("online", "offline")
                if current_stage == "online":
                    self.initial_condition_product = self.assemble_error_estimation_operators(("initial_condition", "initial_condition"), "online")
                elif current_stage == "offline":
                    self.initial_condition_product = OnlineAffineExpansionStorage(self.Q_ic, self.Q_ic)
                else:
                    raise AssertionError("Invalid stage in _init_error_estimation_operators().")
                    
        ## Build operators for error estimation
        def build_error_estimation_operators(self):
            # Call Parent
            TimeDependentRBReducedProblem_Base.build_error_estimation_operators(self)
            # Assemble initial condition product error estimation operator
            if not self.initial_condition_is_homogeneous:
                self.assemble_error_estimation_operators(("initial_condition", "initial_condition"), "offline") 
        
        ## Assemble operators for error estimation
        @override
        def assemble_error_estimation_operators(self, term, current_stage="online"):
            if term[0] == "initial_condition" and term[1] == "initial_condition":
                assert current_stage in ("online", "offline")
                if current_stage == "online": # load from file
                    if self.initial_condition_product is None:
                        self.initial_condition_product = OnlineAffineExpansionStorage(0, 0) # it will be resized by load
                    self.initial_condition_product.load(self.folder["error_estimation"], "initial_condition_product")
                    return self.initial_condition_product
                elif current_stage == "offline":
                    for q in range(self.Q_ic):
                        for qp in range(q, self.Q_ic):
                            self.initial_condition_product[q, qp] = transpose(self.truth_problem.initial_condition[q])*self._riesz_solve_inner_product*self.truth_problem.initial_condition[qp]
                            if q != qp:
                                self.initial_condition_product[qp, q] = self.initial_condition_product[q, qp]
                    self.initial_condition_product.save(self.folder["error_estimation"], "initial_condition_product")
                    return self.initial_condition_product
                else:
                    raise AssertionError("Invalid stage in assemble_error_estimation_operators().")
            else:
                return TimeDependentRBReducedProblem_Base.assemble_error_estimation_operators(self, term, current_stage)
                
    # return value (a class) for the decorator
    return TimeDependentRBReducedProblem_Class
    
