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
## @file parabolic_coercive_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for parabolic coercive problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.backends import ProperOrthogonalDecomposition
from RBniCS.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from RBniCS.reduction_methods.elliptic_coercive import EllipticCoercivePODGalerkinReduction

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoercivePODGalerkinReduction
#
# Base class containing the interface of a POD-Galerkin ROM
# for parabolic coercive problems
@Extends(EllipticCoercivePODGalerkinReduction) # needs to be first in order to override for last the methods
@ReductionMethodFor(ParabolicCoerciveProblem, "PODGalerkin")
class ParabolicCoercivePODGalerkinReduction(EllipticCoercivePODGalerkinReduction):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem):
        # Call the parent initialization
        EllipticCoercivePODGalerkinReduction.__init__(self, truth_problem)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
