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
## @file parabolic_coercive_pod_base.py
#  @brief Implementation of a POD-Galerkin ROM for parabolic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from parabolic_coercive_base import *
from elliptic_coercive_pod_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoercivePODBase
#
# Base class containing the interface of a POD-Galerkin ROM
# for parabolic coercive problems
class ParabolicCoercivePODBase(ParabolicCoerciveBase,EllipticCoercivePODBase):
# Beware of the diamond problem in multiple inheritance: in python precedence is depth-first and then left-to-right
    """This class implements a POD-based approach for parabolic coercive
    problems, assuming the compliance of the output of interest. If
    compared to the ParabolicCoerciveRBBase class, the POD is used for
    both the time and the parameter space.  
    
    """


    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, bc_list):
        # Call the parent initialization
        ParabolicCoerciveBase.__init__(self, V, bc_list)
        EllipticCoercivePODBase.__init__(self, V, bc_list)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Nothing to be added in this case
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{

    ## Update the snapshot matrix
    def update_snapshot_matrix(self):
        self.POD.store_multiple_snapshots(self.all_snap)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 

    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    # Nothing to be added in this case
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
