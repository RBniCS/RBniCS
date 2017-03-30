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

from rbnics.reduction_methods.base import NonlinearReductionMethod
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from rbnics.utils.decorators import Extends, override, MultiLevelReductionMethod

def NavierStokesReductionMethod(StokesReductionMethod_DerivedClass):
    @Extends(StokesReductionMethod_DerivedClass) # needs to be first in order to override for last the methods.
    #@ReductionMethodFor(NavierStokesProblem, "Abstract") # disabled, since now this is a decorator which depends on a derived (e.g. POD or RB) class
    @MultiLevelReductionMethod
    @NonlinearReductionMethod
    class NavierStokesReductionMethod_Class(StokesReductionMethod_DerivedClass):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the reduced order model object
        #  @{
        
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            StokesReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
                
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
    # return value (a class) for the decorator
    return NavierStokesReductionMethod_Class
        
