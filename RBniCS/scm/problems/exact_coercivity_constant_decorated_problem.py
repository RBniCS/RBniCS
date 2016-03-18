# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os # for path and makedir
import shutil # for rm
import glpk # for LB computation
import sys # for sys.float_info.max
import random # to randomize selection in case of equal error bound
import operator # to find closest parameters
from RBniCS.parametrized_problem import ParametrizedProblem

def ExactCoercivityConstantDecoratedProblem(*args):
    def ExactCoercivityConstantDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
        class ExactCoercivityConstantDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            ## Default initialization of members
            def __init__(self, V, *args):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, *args)
                # Attach the exact coercivity constant computation problem
                self.exact_coercivity_constant_computation__problem = _ExactCoercivityConstantComputation_Problem(V) #TODO
                    
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
        
            # Propagate the values of all setters also to the EIM object
            
            ## OFFLINE: set the range of the parameters    
            def set_mu_range(self, mu_range):
                ParametrizedProblem_DerivedClass.set_mu_range(self, mu_range)
                self.exact_coercivity_constant_computation__problem.set_mu_range(mu_range) #TODO
                    
            ## OFFLINE/ONLINE: set the current value of the parameter
            def set_mu(self, mu):
                ParametrizedProblem_DerivedClass.set_mu(self, mu)
                self.exact_coercivity_constant_computation__problem.set_mu(mu) #TODO
                
            #  @}
            ########################### end - SETTERS - end ########################### 
                        
        #~~~~~~~~~~~~~~~~~~~~~~~~~     EXACT COERCIVITY CONSTANT COMPUTATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class _ExactCoercivityConstantComputation_Problem
        #
        # Exact computation of the coercivity constant
        class _ExactCoercivityConstantComputation_Problem(ParametrizedProblem):

            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the exact computation of coercivity constant
            #  @{
        
            ## Default initialization of members
            def __init__(self, parametrized_problem):
                pass # TODO
            
        # return value (a class) for the decorator
        return ExactCoercivityConstantDecoratedProblem_Class
    
    # return the decorator itself
    return ExactCoercivityConstantDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ExactCoercivityConstant = ExactCoercivityConstantDecoratedProblem
    
