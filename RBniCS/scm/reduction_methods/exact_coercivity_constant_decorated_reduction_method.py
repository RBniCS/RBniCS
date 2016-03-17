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

def ExactCoercivityConstantDecoratedReductionMethod(ReductionMethod_DerivedClass):
    class ExactCoercivityConstantDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(truth_problem)
            assert isinstance(truth_problem, ExactCoercivityConstantDecoratedProblem_Class)
            # Attach the exact coercivity constant computation method
            self.exact_coercivity_constant_computation__method = _ExactCoercivityConstantComputation_Method(truth_problem.SCM_approximation) # TODO
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~     EXACT COERCIVITY CONSTANT COMPUTATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class _ExactCoercivityConstantComputation_Method
    #
    # Successive constraint method for the approximation of the coercivity constant
    class _ExactCoercivityConstantComputation_Method(object): # TODO

        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the exact computation of coercivity constant
        #  @{
        
        ## Default initialization of members
        def __init__(self, parametrized_problem):
            pass # TODO
    
