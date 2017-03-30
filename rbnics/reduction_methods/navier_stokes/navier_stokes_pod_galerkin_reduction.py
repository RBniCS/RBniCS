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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.reduction_methods.base import NonlinearPODGalerkinReduction
from RBniCS.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from RBniCS.reduction_methods.stokes import StokesPODGalerkinReduction
from RBniCS.reduction_methods.navier_stokes.navier_stokes_reduction_method import NavierStokesReductionMethod

NavierStokesPODGalerkinReduction_Base = NavierStokesReductionMethod(NonlinearPODGalerkinReduction(StokesPODGalerkinReduction))

@Extends(NavierStokesPODGalerkinReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(NavierStokesProblem, "PODGalerkin")
class NavierStokesPODGalerkinReduction(NavierStokesPODGalerkinReduction_Base):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        NavierStokesPODGalerkinReduction_Base.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
