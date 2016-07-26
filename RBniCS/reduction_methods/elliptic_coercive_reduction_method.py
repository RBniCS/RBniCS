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
## @file elliptic_coercive_reduction_method.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path and makedir
from RBniCS.reduction_methods.reduction_method import ReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReductionMethodBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
class EllipticCoerciveReductionMethod(ReductionMethod):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    from RBniCS.factories.problems import ReducedProblemFactory as ReducedProblemFactory_Function
    ReducedProblemFactory = staticmethod(ReducedProblemFactory_Function)
    # In this way we allow the final user to possibly override ReducedProblemFactory
    
    ## Default initialization of members
    def __init__(self, truth_problem):
        # Call to parent
        ReductionMethod.__init__(self, truth_problem.name(), truth_problem.mu_range)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Reduced order problem
        self.reduced_problem = None
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
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
        self.reduced_problem = self.ReducedProblemFactory(self.truth_problem, self)
        
        # Prepare folders and init reduced problem
        all_folders_exist = True
        all_folders = list()
        all_folders.extend(self.folder.values())
        all_folders.extend(self.reduced_problem.folder.values())
        for f in all_folders:
            if os.path.exists(f) and len(os.listdir(f)) == 0: # already created, but empty
                all_folders_exist = False
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
