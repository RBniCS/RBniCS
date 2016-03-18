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
## @file elliptic_coercive_reduction_method_base.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os # for path and makedir

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReductionMethodBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
class EllipticCoerciveReductionMethodBase(ReductionMethodBase):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, truth_problem):
        # Call to parent
        ReductionMethodBase.__init__(self, truth_problem.name())
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 3. Reduced order problem
        self.reduced_problem = None
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3. High fidelity problem
        self.truth_problem = truth_problem
            
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    def _init_offline(self):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem = ReducedProblemFactory(self.truth_problem, self)
        
        # Since we set the initial parameter before the call to offline(),
        # we also need to sync the reduced problem's mu with the
        # truth problem's mu (user prescribed)
        self.reduced_problem.set_mu(self.truth_problem.mu)
        # Note that, instead, setting the reduced problem current value of mu
        # gets automatically propagated to the truth problem.
        
        # Prepare folders and init reduced problem
        all_folders_exist = True
        for f in self.folder.values():
            if not os.path.exists(f):
                all_folders_exist = False
                os.makedirs(f)
        if all_folders_exist:
            self.reduced_problem.init("online")
            return False # offline construction should be skipped, since data are already available
        else:
            self.reduced_problem.init("offline")
            return True # offline construction should be carried out
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    ## Initialize data structures required for the error analysis phase
    def _init_error_analysis(self):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem.init("online")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
