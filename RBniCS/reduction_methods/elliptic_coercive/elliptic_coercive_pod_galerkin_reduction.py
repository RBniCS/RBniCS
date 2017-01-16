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
## @file elliptic_coercive_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods.base import PODGalerkinReduction
from RBniCS.reduction_methods.elliptic_coercive.elliptic_coercive_reduction_method import EllipticCoerciveReductionMethod
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoercivePODGalerkinReduction
#

EllipticCoercivePODGalerkinReduction_Base = PODGalerkinReduction(EllipticCoerciveReductionMethod)

# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@Extends(EllipticCoercivePODGalerkinReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticCoerciveProblem, "PODGalerkin")
class EllipticCoercivePODGalerkinReduction(EllipticCoercivePODGalerkinReduction_Base):
    """This class implements a reduced order method based on a POD (Proper
    Orthogonal Decomposition) Galerkin approach. In particular, it
    implements the offline phase and the error analysis proper for the
    POD approach.
    
    This class provides the following methods:
    
    ##  Methods related to the offline stage
    - offline()
    - update_snapshot_matrix()
    - apply_POD()

    ## Error analysis
    - error_analysis()

    A typical usage of this class is reported in tutorial 2.

    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        EllipticCoercivePODGalerkinReduction_Base.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the testing set
    @override
    def error_analysis(self, N=None, **kwargs):
        if N is None:
            N = self.reduced_problem.N
            
        EllipticCoercivePODGalerkinReduction_Base.error_analysis(self, N, **kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
