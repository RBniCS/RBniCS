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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override

def TimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class TimeDependentRBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
                
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the reduced order model object
        #  @{
        
        ## Default initialization of members.
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Storage related to error estimation for initial condition
            self.initial_condition_product = None # will be of class OnlineAffineExpansionStorage

            
        #  @}
        ########################### end - CONSTRUCTORS - end ########################### 
        
        ###########################     ONLINE STAGE     ########################### 
        ## @defgroup OnlineStage Methods related to the online stage
        #  @{
        
        def _init_error_estimation_operators(self, current_stage="online"):
            ParametrizedReducedDifferentialProblem_DerivedClass._init_error_estimation_operators(self, current_stage)
            # Also initialize data structures related to initial condition error estimation
            if not self.initial_condition_is_homogeneous:
                assert current_stage in ("online", "offline")
                if current_stage == "online":
                    self.initial_condition_product = self.assemble_error_estimation_operators("initial_condition_product", "online")
                elif current_stage == "offline":
                    self.initial_condition_product = OnlineAffineExpansionStorage(self.Q_ic, self.Q_ic)
                else:
                    raise AssertionError("Invalid stage in _init_error_estimation_operators().")
                                
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
                
    # return value (a class) for the decorator
    return TimeDependentRBReducedProblem_Class
    
