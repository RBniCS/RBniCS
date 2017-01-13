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
## @file elliptic_coercive_rb.py
#  @brief Implementation of the reduced basis method for (compliant) elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods.base import RBReduction
from RBniCS.reduction_methods.elliptic_coercive.elliptic_coercive_reduction_method import EllipticCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRB
#

EllipticCoerciveRBReduction_Base = RBReduction(EllipticCoerciveReductionMethod)

# Base class containing the interface of the RB method
# for (compliant) elliptic coercive problems
@Extends(EllipticCoerciveRBReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticCoerciveProblem, "ReducedBasis")
class EllipticCoerciveRBReduction(EllipticCoerciveRBReduction_Base):
    """This class implements the Certified Reduced Basis Method for
    elliptic and coercive problems. The output of interest are assumed to
    be compliant.

    During the offline stage, the parameters are chosen relying on a
    greedy algorithm. The user must specify how the alpha_lb (i.e., alpha
    lower bound) is computed since this term is needed in the a posteriori
    error estimation. RBniCS features an implementation of the Successive
    Constraints Method (SCM) for the estimation of the alpha_lb (take a
    look at tutorial 4 for the usage of SCM).
    
    The following functions are implemented:

    ## Methods related to the offline stage
    - offline()
    - update_basis_matrix()
    - greedy()
    - compute_dual_terms()
    - compute_a_dual()
    - compute_f_dual()

    ## Methods related to the online stage
    - online_output()
    - estimate_error()
    - estimate_error_output()
    - truth_output()

    ## Error analysis
    - compute_error()
    - error_analysis()
    
    ## Input/output methods
    - load_reduced_matrices()
    
    ## Problem specific methods
    - get_alpha_lb() # to be overridden

    A typical usage of this class is given in the tutorial 1.

    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        EllipticCoerciveRBReduction_Base.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        def solve_and_estimate_error(mu, index):
            self.reduced_problem.set_mu(mu)
            self.reduced_problem._solve(self.reduced_problem.N)
            return self.reduced_problem.estimate_error()
            
        return self.training_set.max(solve_and_estimate_error)

    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the testing set
    @override
    def error_analysis(self, N=None, **kwargs):
        if N is None:
            N = self.reduced_problem.N
            
        EllipticCoerciveRBReduction_Base.error_analysis(self, N, **kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
        
